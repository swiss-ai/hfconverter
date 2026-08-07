# Copyright 2026 the Swiss AI Initiative. All rights reserved.
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
"""Megatron torch_dist to Hugging Face safetensors exporter for Apertus 2 MoE.

Heavy imports (torch, megatron.core, transformers) live in the submodules; importing the
bare package stays cheap. Entry points: ``python -m exporter.export`` (CLI) and
``exporter.export.export_checkpoint`` (python API).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
