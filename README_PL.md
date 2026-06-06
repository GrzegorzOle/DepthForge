# DepthForge

> **Odczyt głębi z obrazów 2D — dotykowe reprodukcje 3D dla osób niewidomych**

![Porównanie map głębokości – Józef Chełmoński „Babie lato"](assets/depth_comparison_preview.jpg)
*Porównanie map głębokości dla obrazu Józefa Chełmońskiego „Babie lato" (Google Art Project).  
Od lewej: oryginał · Standard syntetyczny · OpenVINO MiDaS v2.1 Small · OpenVINO DPT Large · Ensemble (fuzja wszystkich metod)*

---

## Przeznaczenie

DepthForge analizuje płaski obraz 2D i rekonstruuje ukrytą w nim **informację o głębi (przestrzeni)** — szacując, które fragmenty sceny są bliskie, a które odległe.  
Wygenerowane mapy głębokości stanowią podstawę do tworzenia **dotykowych reprodukcji 3D**, które umożliwiają osobom niewidomym fizyczne poznanie dzieł sztuki i eksponatów muzealnych poprzez dotyk.

Proces przebiega dwuetapowo:

1. **DepthForge** — automatyczna ekstrakcja głębi z obrazu przy użyciu modeli AI (MiDaS, DPT) przyspieszonych przez Intel OpenVINO.  
2. **Przygotowanie do druku 3D** — dane o głębi trafiają do specjalisty, który przygotowuje je w odpowiedniej formie fizycznej umożliwiającej eksplorację dotykową.  
   Za ten etap odpowiada **Jakub Oleksy**, specjalista ds. analizy druku 3D:  
   [linkedin.com/in/jakub-oleksy-672668333/](https://www.linkedin.com/in/jakub-oleksy-672668333/)

> **Wtyczka GIMP** — repozytorium zawiera również wstępną, zalążkową warstwę integracji z edytorem graficznym GIMP (`src/gimp_plugin.py`). Wtyczka **nie jest jeszcze funkcjonalna** — trwają prace integracyjne.

---

## Opis

DepthForge to narzędzie do generowania map głębokości z obrazów muzealnych, przeznaczone do tworzenia dotykowych wizualizacji 3D dla osób niewidomych.

Projekt umożliwia przetwarzanie obrazów muzealnych w celu wygenerowania map głębokości, które mogą być wykorzystane do tworzenia dotykowych map 3D dla osób niewidomych. Wykorzystuje:
- OpenCV do operacji na obrazach
- OpenVINO do efektywnego przetwarzania modeli ML
- PyTorch do analizy obrazów
- Specjalistyczne algorytmy do lepszego wizualizowania głębokości

## Wymagania

- Python 3.8+
- OpenCV
- OpenVINO
- PyTorch
- NumPy
- SciPy
- Scikit-image
- numpy-stl
- transformers

## Instalacja

```bash
# Sklonuj repozytorium
git clone <repo_url>
cd DepthForge

# Utwórz środowisko wirtualne
python -m venv .venv
source .venv/bin/activate

# Zainstaluj wymagane biblioteki
pip install -r requirements.txt
```

## Użycie

### Jedno zdjęcie

```bash
python src/depth_forge.py --input input_image.jpg --output output_depth.png --enhanced-output enhanced_depth.png --tactile-output tactile_map.png
```

### Pełny pipeline (mapy głębokości + STL do druku 3D)

```bash
python src/depth_pipeline.py --input data/Stańczyk.jpg --output-dir output/stanczyk --width-mm 200 --relief-mm 12
```

### Przetwarzanie wsadowe

```bash
python src/depth_forge.py --batch --input-dir data/ --output-dir output/
```

### Benchmark (wszystkie metody + ensemble)

```bash
python benchmark.py
```

## Funkcjonalności specyficzne dla muzeów

- Generowanie map głębokości dla obrazów muzealnych
- Optymalizacja dla wizualizacji 3D
- Wsparcie dla druku 3D (mapy dotykowe) — eksport binarny STL
- Współpraca z systemami brajla i wizualizacji 3D

## Wersje map głębokości

Projekt generuje trzy różne wersje mapy głębokości:
1. **Podstawowa mapa głębokości** — wygenerowana na podstawie intensywności obrazu
2. **Ulepszona mapa głębokości** — zastosowane techniki kontrastu (CLAHE)
3. **Mapa dotykowa** — optymalizowana do druku 3D dla osób niewidomych

## Integracja z OpenVINO

Ten projekt został zaprojektowany z myślą o integracji z OpenVINO w celu poprawy estymacji głębokości:
- Wsparcie dla modeli MiDaS i DPT
- Efektywne wnioskowanie na CPU/GPU
- Integracja z procesami druku 3D

## Struktura projektu

```
DepthForge/
├── assets/              # Zasoby statyczne (obrazy podglądowe itp.)
├── config.json          # Konfiguracja projektu
├── requirements.txt     # Wymagane biblioteki
├── benchmark.py         # Benchmark — wszystkie metody + ensemble
├── src/                 # Kod źródłowy
│   ├── depth_forge.py   # Główny moduł generowania map głębokości
│   ├── depth_pipeline.py# Pełny pipeline: głębokość → STL
│   ├── advanced_3d_generator.py
│   └── gimp_plugin.py   # Integracja z GIMP (w trakcie opracowania)
├── data/                # Katalog danych wejściowych
├── models/              # Modele ML (MiDaS i DPT w formacie OpenVINO)
│   ├── midas/openvino/
│   └── dpt/openvino/
└── output/              # Katalog danych wyjściowych
```

## Konfiguracja

Konfiguracja znajduje się w pliku `config.json`:
- `model.depth_estimation`: Ustawienia modelu estymacji głębokości
- `processing`: Ustawienia przetwarzania obrazów
- `tactile`: Ustawienia wizualizacji dotykowej

## Dla użytkowników zainteresowanych wizualizacjami 3D

Ten projekt jest zaprojektowany do tworzenia map głębokości, które mogą być wykorzystane do:
1. Tworzenia dotykowych map 3D dla osób niewidomych
2. Wizualizacji obrazów muzealnych w formie dotykalnej
3. Integracji z systemami brajla i wizualizacji 3D

## Rozwój

Projekt może zostać rozbudowany o:
- Integrację z konkretnymi modelami głębokości (MiDaS, DPT)
- Wsparcie dla różnych formatów obrazów muzealnych
- Interfejs graficzny
- Obsługę kamer internetowych
- Integrację z systemami druku 3D
