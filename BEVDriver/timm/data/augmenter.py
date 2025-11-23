import logging

try:
    import imgaug as ia  # type: ignore
    from imgaug import augmenters as iaa  # type: ignore
except Exception as exc:  # pragma: no cover - defensive import
    ia = None
    iaa = None
    logging.getLogger(__name__).warning(
        "imgaug import failed in augmenter (%s); image augmentations will be disabled.",
        exc,
    )

def augment(prob=0.2):
    """
    Build an imgaug augmenter. If imgaug is not available, return a no-op
    callable so that downstream code can continue to run without augmentations.
    """
    if iaa is None:
        def _noop(image):
            return image
        return _noop

    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(prob, iaa.GaussianBlur((0, 0.5))),
            iaa.Sometimes(
                prob,
                iaa.AdditiveGaussianNoise(
                    loc=0,
                    scale=(0.0, 0.05 * 255),
                    per_channel=0.5,
                ),
            ),
            iaa.Sometimes(prob, iaa.Dropout((0.01, 0.1), per_channel=0.5)),
            iaa.Sometimes(prob, iaa.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
            iaa.Sometimes(
                prob,
                iaa.LinearContrast((1 / 1.2, 1.2), per_channel=0.5),
            ),
            iaa.Sometimes(prob, iaa.Grayscale((0.0, 1))),
            iaa.Sometimes(
                prob,
                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
            ),
        ],
        random_order=True,
    )
    return augmenter
