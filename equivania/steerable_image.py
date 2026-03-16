import torch
import torchvision.transforms.functional as TF


class SteerableImage:
    r"""
    A steerable image that can be projected on the image grid at an arbitrary orientation

    :param str impl: The implementation to use for steerability
    :param float base_orientation: Base orientation offset (degrees) applied to all steering
    :param torch.Tensor image: (rotate impl) the already rasterized image to rotate
    """

    def __init__(
        self,
        *,
        impl: str = "gabor",
        image: torch.Tensor | None = None,
        base_orientation: float = 112.5,
    ) -> None:
        self._impl = impl
        self._image = image
        self._base_orientation = base_orientation

    def steer(self, angle: float) -> torch.Tensor:
        r"""
        Project a steerable image onto the image grid for a given orientation

        :param float angle: The orientation of the steerable image
        :return: torch.Tensor, the steerable image projected onto the image grid
        """
        return self._steer(
            angle,
            impl=self._impl,
            image=self._image,
            base_orientation=self._base_orientation,
        )

    @staticmethod
    def _steer(
        angle: float,
        *,
        impl: str = "rotate",
        image: torch.Tensor | None = None,
        base_orientation: float = 112.5,
    ) -> torch.Tensor:
        r"""
        Project a steerable image onto the image grid for a given orientation

        :param float angle: The orientation of the steerable image
        :param str impl: The implementation to use. Currently only "gabor" is supported
        :param float base_orientation: Base orientation offset (degrees) applied to all steering
        :param torch.Tensor image: (rotate impl) the already rasterized image to rotate
        :return: torch.Tensor, the steerable image projected onto the image grid
        """
        if impl == "rotate":
            if image is None:
                raise ValueError("image must be provided when using impl='rotate'")
            rotated_angle = (base_orientation + angle) % 360
            x = TF.rotate(
                image,
                rotated_angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                expand=False,
                fill=0,
            )
        elif impl == "gabor":
            from gabor import gabor_image

            theta = (base_orientation + angle) % 360
            x = gabor_image(512, 512, theta=theta, seed=0)
        else:
            raise ValueError(f"Unknown impl: {impl}")

        return x
