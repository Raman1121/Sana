# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

TRANSFORMS = dict()


def register_transform(transform):
    name = transform.__name__
    if name in TRANSFORMS:
        raise RuntimeError(f"Transform {name} has already registered.")
    TRANSFORMS.update({name: transform})


def get_transform(type, resolution):
    transform = TRANSFORMS[type](resolution)
    transform = T.Compose(transform)
    transform.image_size = resolution
    return transform


@register_transform
def default_train(n_px):
    transform = [
        T.Lambda(lambda img: img.convert("RGB")),
        T.RandomResizedCrop(
            n_px,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BICUBIC
        ),
        # T.CenterCrop(n_px),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transform

@register_transform
def default_train2(n_px):
    print("Using default_train2 transform for X-ray images.")
    transform = [
        T.Lambda(lambda img: img.convert("RGB")),

        # Use RandomResizedCrop with conservative parameters.
        # This acts as a 'scale and slight translation' augmentation.
        # It will always crop a region close to the center but not the *exact* same one.
        # scale=(0.9, 1.0): Only zooms in by at most 10%.
        # ratio=(0.95, 1.05): Keeps the aspect ratio almost square.
        T.RandomResizedCrop(
            n_px,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BICUBIC
        ),

        # Horizontal flipping is generally safe for X-rays as anatomy is
        # largely symmetrical. It's a powerful augmentation.
        # (Verify this assumption for your specific task/pathologies).
        T.RandomHorizontalFlip(p=0.5),

        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    return transform