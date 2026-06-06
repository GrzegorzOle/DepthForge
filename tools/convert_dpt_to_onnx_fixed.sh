#!/bin/bash

# Konwersja modelu DPT z PyTorch do ONNX
# Plik: convert_dpt_to_onnx.sh

echo "Konwersja modelu DPT do formatu ONNX..."
echo "======================================"

# Sprawdzenie czy potrzebne narzędzia są zainstalowane
echo "Sprawdzanie wymagań..."
if ! command -v python &> /dev/null; then
    echo "Błąd: Python nie jest zainstalowany"
    exit 1
fi

# Sprawdzenie czy biblioteki są dostępne
echo "Sprawdzanie bibliotek..."
python -c "import torch; import transformers; print('Wymagane biblioteki dostępne')" 2>/dev/null || {
    echo "Błąd: Brak wymaganych bibliotek"
    echo "Zainstaluj: pip install torch transformers"
    exit 1
}

# Sprawdzenie czy plik modelu istnieje
if [ ! -f "models/dpt/pytorch_model.bin" ]; then
    echo "Błąd: Nie znaleziono pliku modelu pytorch_model.bin"
    echo "Upewnij się, że model DPT został pobrany"
    exit 1
fi

# Wykonanie konwersji
echo "Uruchamianie konwersji modelu..."
python -c "
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
import onnx
import os

print('Ładowanie modelu DPT...')
# Wczytaj model i procesor
model = DPTForDepthEstimation.from_pretrained('models/dpt')
processor = DPTImageProcessor.from_pretrained('models/dpt')

print('Tworzenie dummy input...')
# Tworzenie dummy input dla konwersji
dummy_input = torch.randn(1, 3, 384, 384)

print('Konwersja do ONNX...')
# Konwersja do ONNX
onnx_model_path = 'models/dpt/dpt_large.onnx'
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 1: 'height', 2: 'width'}
    }
)

print('Konwersja zakończona!')
print(f'Model zapisany jako: {onnx_model_path}')
print('')
print('Możesz teraz skonwertować do formatu OpenVINO:')
print('mo --input_model models/dpt/dpt_large.onnx --output_dir models/dpt --model_name dpt_depth --compress_to_fp16')
"

echo "======================================"
echo "Konwersja zakończona!"
echo ""
echo "Następny krok - konwersja do OpenVINO:"
echo "mo --input_model models/dpt/dpt_large.onnx --output_dir models/dpt --model_name dpt_depth --compress_to_fp16"