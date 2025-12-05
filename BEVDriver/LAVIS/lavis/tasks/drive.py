"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os

import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks.base_task import BaseTask


@registry.register_task("carla_drive")
class DriveTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        output = model(samples)
        return output

    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, val_result, epoch, writer=None, **kwargs):
        output = {}
        nums = 0
        for each in val_result:
            nums += 1
            for key in each:
                if key not in output:
                    output[key] = 0
                output[key] += each[key]
        for key in output:
            if isinstance(output[key], float) or isinstance(output[key], int):
                output[key] = output[key]/nums
            else:
                output[key] = output[key].item()/nums

        out_str = ''
        for key in output:
            out_str += '%s: %.3f, ' % (key, output[key])
            if is_main_process():
                writer.add_scalar('val/%s_epoch' % key, output[key], epoch)
        logging.info('Eval Epoch %d, %s' % (epoch, out_str))
        output['agg_metrics'] = output['loss']

        return output

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        # Optional: limit number of validation iterations for quick debugging.
        cfg = registry.get("configuration", default=None, no_warning=True)
        if cfg is not None:
            max_val_iters = cfg.run_cfg.get("debug_max_val_iters", None)
            if max_val_iters is not None:

                class _LimitedLoader:
                    def __init__(self, loader, max_iters):
                        self.loader = loader
                        self.max_iters = int(max_iters)

                    def __iter__(self):
                        for i, batch in enumerate(self.loader):
                            if i >= self.max_iters:
                                break
                            yield batch

                    def __len__(self):
                        try:
                            base_len = len(self.loader)
                        except TypeError:
                            base_len = self.max_iters
                        return min(base_len, self.max_iters)

                data_loader = _LimitedLoader(data_loader, max_val_iters)

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            with torch.cuda.amp.autocast(enabled=True):
                eval_output = self.valid_step(model=model, samples=samples)
            results.append(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results


@registry.register_task("bevqa_drive")
class BEVQADriveTask(DriveTask):
    """
    DriveTask variant for BEV-QA that, in addition to loss, runs model.generate()
    on validation samples and dumps {id, pred, ref} JSON for external metrics
    (e.g., SPICE wrapper).
    """

    def __init__(
        self,
        num_beams: int = 1,
        max_new_tokens: int = 32,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        skip_generate: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.skip_generate = skip_generate
        self.min_new_tokens = min_new_tokens

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            num_beams=run_cfg.get("num_beams", 1),
            max_new_tokens=run_cfg.get("max_new_tokens", 32),
            top_p=run_cfg.get("top_p", 0.9),
            temperature=run_cfg.get("temperature", 1.0),
            repetition_penalty=run_cfg.get("repetition_penalty", 1.0),
            min_new_tokens=run_cfg.get("min_new_tokens", 0),
            skip_generate=run_cfg.get("skip_generate", False),
        )

    def valid_step(self, model, samples):
        # Compute loss
        output = model(samples)

        # Optionally skip generation (for unstable HF/PEFT combos or faster val).
        if self.skip_generate:
            return {"loss": output["loss"]}

        preds = model.generate(
            samples,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            min_new_tokens=self.min_new_tokens,
        )

        refs = samples["vqa_answer"]
        ids = samples.get("id", None)

        # ids may be a list of strings or tensors; keep as-is, normalize later.
        return {
            "loss": output["loss"],
            "pred": preds,
            "ref": refs,
            "id": ids,
        }

    def after_evaluation(self, val_result, epoch, split_name=None, writer=None, **kwargs):
        # Aggregate numeric metrics (e.g., loss) while preserving predictions.
        metrics = {}
        nums = 0

        for each in val_result:
            nums += 1
            for key, value in each.items():
                if key in ("pred", "ref", "id"):
                    continue
                if key not in metrics:
                    metrics[key] = 0.0
                if isinstance(value, (float, int)):
                    metrics[key] += float(value)
                else:
                    metrics[key] += float(value.detach().cpu().item())

        if nums > 0:
            for key in metrics:
                metrics[key] /= nums

        out_str = ""
        for key in metrics:
            out_str += "%s: %.3f, " % (key, metrics[key])
            if writer is not None and is_main_process():
                writer.add_scalar("val/%s_epoch" % key, metrics[key], epoch)

        if out_str:
            # Local change: use string placeholder to allow non-integer epoch labels (e.g., "best").
            logging.info("Eval Epoch %s, %s", epoch, out_str)

        if "loss" in metrics:
            metrics["agg_metrics"] = metrics["loss"]

        # Collect predictions and references into a flat list.
        if is_main_process():
            records = []
            for batch_out in val_result:
                batch_ids = batch_out.get("id", [])
                batch_preds = batch_out.get("pred", [])
                batch_refs = batch_out.get("ref", [])

                # Normalize ids to Python strings.
                if isinstance(batch_ids, torch.Tensor):
                    batch_ids = batch_ids.cpu().tolist()

                for sid, pred, ref in zip(batch_ids, batch_preds, batch_refs):
                    if isinstance(sid, torch.Tensor):
                        sid = sid.item()
                    sid = str(sid)
                    records.append(
                        {
                            "id": sid,
                            "pred": pred,
                            "ref": ref,
                        }
                    )

            out_dir = registry.get_path("output_dir")
            os.makedirs(out_dir, exist_ok=True)
            prefix = split_name if split_name is not None else "val"
            out_path = os.path.join(out_dir, f"{prefix}_bevqa_epoch{epoch}.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            logging.info("Saved BEV-QA predictions to %s", out_path)

        return metrics

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        writer=None,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            writer=writer,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        writer=None,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        # Optional: limit number of training iterations per epoch for quick debugging.
        cfg = registry.get("configuration", default=None, no_warning=True)
        if cfg is not None:
            max_train_iters = cfg.run_cfg.get("debug_max_train_iters", None)
            if max_train_iters is not None:
                iters_per_epoch = min(iters_per_epoch, int(max_train_iters))

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            if is_main_process() and writer is not None:
                for key in loss_dict:
                    value = loss_dict[key]
                    if isinstance(value, (float, int)):
                        writer.add_scalar('train/%s_iter' % key, value, epoch*iters_per_epoch+i)
                    elif torch.is_tensor(value):
                        if value.numel() == 1:
                            writer.add_scalar('train/%s_iter' % key, value.item(), epoch*iters_per_epoch+i)
                    # skip non-scalar tensors

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if is_main_process():
            for key, meter in metric_logger.meters.items():
                writer.add_scalar('train/%s_epoch' % key, meter.global_avg, epoch)
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
