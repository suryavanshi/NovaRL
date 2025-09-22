"""Vision-language prompt environment with multi-modal preprocessing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from core.interfaces import EnvStep
from envs.base import BatchedEnvironment


# ---------------------------------------------------------------------------
# Dataset containers
# ---------------------------------------------------------------------------


@dataclass
class VisionPromptSample:
    """Single prompt consisting of an image, a question, and answer choices."""

    image: np.ndarray | torch.Tensor
    question: str
    choices: Sequence[str]
    answer_index: int
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("VisionPromptSample requires at least one answer choice")
        if not 0 <= self.answer_index < len(self.choices):
            raise ValueError(
                "answer_index must reference one of the provided answer choices"
            )


class VisionPromptDataset(Sequence[VisionPromptSample]):
    """In-memory dataset of vision-language prompts."""

    def __init__(self, samples: Sequence[VisionPromptSample]) -> None:
        if not samples:
            raise ValueError("VisionPromptDataset requires at least one sample")
        choice_sizes = {len(sample.choices) for sample in samples}
        if len(choice_sizes) != 1:
            raise ValueError("All samples must expose the same number of answer choices")
        self._samples = list(samples)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __getitem__(self, index: int) -> VisionPromptSample:
        return self._samples[index]

    @property
    def action_dim(self) -> int:
        """Number of discrete answers available for each sample."""

        return len(self._samples[0].choices)

    def iter_texts(self) -> Iterable[str]:
        """Yield every textual fragment for vocabulary building utilities."""

        for sample in self._samples:
            yield sample.question
            for choice in sample.choices:
                yield choice


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------
def _to_chw_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert supported inputs into a 3xHxW floating point tensor."""

    if isinstance(data, torch.Tensor):
        tensor = data.detach().clone().float()
    else:
        tensor = torch.from_numpy(np.asarray(data)).float()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
        pass
    elif tensor.ndim == 3 and tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(2, 0, 1)
    else:  # pragma: no cover - defensive branch
        raise ValueError("Unsupported image tensor shape")

    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, *tensor.shape[1:])
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def default_image_transform(image: np.ndarray | torch.Tensor, size: int = 64) -> torch.Tensor:
    """Resize and normalise the image to a square float tensor."""

    tensor = _to_chw_tensor(image)
    if tensor.shape[1:] != (size, size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


# ---------------------------------------------------------------------------
# Lightweight text vectoriser for embedding-style collators
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class SimpleTextVectoriser:
    """Bag-of-words style vectoriser for toy embedding models."""

    def __init__(self, vocabulary: Sequence[str]) -> None:
        if not vocabulary:
            raise ValueError("Vocabulary must contain at least one token")
        self._token_to_index = {token: idx for idx, token in enumerate(vocabulary)}

    @classmethod
    def from_dataset(cls, dataset: VisionPromptDataset) -> "SimpleTextVectoriser":
        tokens: set[str] = set()
        for text in dataset.iter_texts():
            tokens.update(_tokenise(text))
        if not tokens:
            tokens = {"placeholder"}
        return cls(sorted(tokens))

    @property
    def dimension(self) -> int:
        return len(self._token_to_index)

    def encode(self, text: str) -> torch.Tensor:
        counts = torch.zeros(self.dimension, dtype=torch.float32)
        for token in _tokenise(text):
            idx = self._token_to_index.get(token)
            if idx is not None:
                counts[idx] += 1.0
        norm = torch.linalg.norm(counts)
        if norm > 0:
            counts /= norm
        return counts


# ---------------------------------------------------------------------------
# Multi-modal collators
# ---------------------------------------------------------------------------


class VisionPromptEmbeddingCollator:
    """Produce fused text and image embeddings for tiny VLM backbones."""

    def __init__(
        self,
        dataset: VisionPromptDataset,
        *,
        text_vectoriser: SimpleTextVectoriser | None = None,
        image_transform: Callable[[np.ndarray | torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.dataset = dataset
        self.text_vectoriser = text_vectoriser or SimpleTextVectoriser.from_dataset(dataset)
        self.image_transform = image_transform or (lambda image: default_image_transform(image))

    def __call__(self, indices: Sequence[int]) -> Mapping[str, torch.Tensor]:
        text_features = []
        image_features = []
        labels = []
        for index in indices:
            sample = self.dataset[index]
            combined_text = " ".join([sample.question, *sample.choices])
            text_features.append(self.text_vectoriser.encode(combined_text))
            image_tensor = self.image_transform(sample.image)
            image_features.append(image_tensor.reshape(-1))
            labels.append(sample.answer_index)
        text_tensor = torch.stack(text_features, dim=0)
        image_tensor = torch.stack(image_features, dim=0)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return {
            "text_embeddings": text_tensor,
            "image_embeddings": image_tensor,
            "labels": label_tensor,
        }


class VisionPromptTokenCollator:
    """Collate prompts for models that rely on special image tokens."""

    def __init__(
        self,
        dataset: VisionPromptDataset,
        tokenizer,
        *,
        image_transform: Callable[[np.ndarray | torch.Tensor], torch.Tensor] | None = None,
        image_token: str = "<image>",
        chat_formatter: Callable[[VisionPromptSample, str], str] | None = None,
    ) -> None:
        try:
            vocab_size_before = len(tokenizer)
        except TypeError:  # pragma: no cover - some tokenisers do not define __len__
            vocab_size_before = None
        added = 0
        if hasattr(tokenizer, "add_special_tokens"):
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": [image_token]}
            )
        if added == 0 and vocab_size_before is not None and image_token not in tokenizer.get_vocab():
            raise ValueError(
                "Tokenizer must recognise the image token. Ensure it is added as an additional special token."
            )
        token_id = tokenizer.convert_tokens_to_ids(image_token)
        if token_id is None or token_id < 0:
            raise ValueError("Tokenizer failed to provide an id for the image token")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform or (lambda image: default_image_transform(image))
        self.image_token = image_token
        self.image_token_id = int(token_id)
        self.chat_formatter = chat_formatter

    def _format_prompt(self, sample: VisionPromptSample) -> str:
        if self.chat_formatter is not None:
            return self.chat_formatter(sample, self.image_token)
        options = []
        for idx, choice in enumerate(sample.choices):
            label = chr(ord("A") + idx)
            options.append(f"({label}) {choice}")
        option_block = "\n".join(options)
        return (
            f"{self.image_token}\n"
            f"Question: {sample.question}\n"
            f"Choices:\n{option_block}\n"
            "Answer:"
        )

    def __call__(self, indices: Sequence[int]) -> Mapping[str, torch.Tensor]:
        prompts = []
        images = []
        labels = []
        for index in indices:
            sample = self.dataset[index]
            prompts.append(self._format_prompt(sample))
            images.append(self.image_transform(sample.image))
            labels.append(sample.answer_index)
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        pixel_values = torch.stack(images, dim=0)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": pixel_values,
            "labels": label_tensor,
        }


# ---------------------------------------------------------------------------
# Environment definition
# ---------------------------------------------------------------------------


class VisionPromptEnvironment(BatchedEnvironment):
    """Environment that surfaces multimodal prompts for VQA-style tasks."""

    def __init__(
        self,
        dataset: VisionPromptDataset,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size, device=device)
        self.dataset = dataset
        self._answers = torch.tensor(
            [sample.answer_index for sample in dataset],
            dtype=torch.long,
            device=self.device,
        )
        self.current_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._sample_indices()

    @property
    def num_prompts(self) -> int:
        return len(self.dataset)

    @property
    def action_dim(self) -> int:
        return self.dataset.action_dim

    def reset(self, batch_size: int | None = None) -> EnvStep:
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError("VisionPromptEnvironment has fixed batch size")
        self._sample_indices()
        observations = self.current_indices.clone().unsqueeze(-1)
        zeros = torch.zeros(self.batch_size, device=self.device)
        infos: Sequence[MutableMapping[str, object]] = tuple(
            self._make_info(int(idx), None, None) for idx in self.current_indices
        )
        return EnvStep(observations=observations, rewards=zeros, dones=zeros, infos=infos)

    def step(self, actions: torch.Tensor) -> EnvStep:
        actions = actions.long().view(-1)
        if actions.shape[0] != self.batch_size:
            raise ValueError("Action batch size mismatch in VisionPromptEnvironment")
        prompt_indices = self.current_indices
        if torch.any(actions < 0) or torch.any(actions >= self.action_dim):
            raise ValueError("Action index out of bounds for VisionPromptEnvironment")
        correct_answers = self._answers[prompt_indices]
        rewards = (actions == correct_answers).float()
        dones = torch.ones(self.batch_size, device=self.device)
        infos: Sequence[MutableMapping[str, object]] = tuple(
            self._make_info(int(p.item()), int(a.item()), bool(r.item()))
            for p, a, r in zip(prompt_indices, actions, rewards)
        )
        self._sample_indices()
        next_obs = self.current_indices.clone().unsqueeze(-1)
        return EnvStep(observations=next_obs, rewards=rewards, dones=dones, infos=infos)

    def _sample_indices(self) -> None:
        self.current_indices = torch.randint(
            low=0,
            high=self.num_prompts,
            size=(self.batch_size,),
            device=self.device,
        )

    def _make_info(
        self, prompt_index: int, action: int | None, correct: bool | None
    ) -> MutableMapping[str, object]:
        sample = self.dataset[prompt_index]
        info: MutableMapping[str, object] = {
            "prompt_index": prompt_index,
            "question": sample.question,
            "choices": tuple(sample.choices),
            "answer_index": sample.answer_index,
        }
        if action is not None:
            info["action"] = action
        if correct is not None:
            info["correct"] = correct
        if sample.metadata:
            info.update(dict(sample.metadata))
        return info


__all__ = [
    "VisionPromptSample",
    "VisionPromptDataset",
    "VisionPromptEnvironment",
    "VisionPromptEmbeddingCollator",
    "VisionPromptTokenCollator",
    "SimpleTextVectoriser",
    "default_image_transform",
]

