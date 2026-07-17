# DepthForge

> **Odczyt głębi z obrazów 2D — dotykowe reprodukcje 3D dla osób niewidomych**

![Porównanie map głębokości – Józef Chełmoński „Babie lato"](assets/depth_comparison_preview.jpg)
*Porównanie map głębokości dla obrazu Józefa Chełmońskiego „Babie lato" (Google Art Project).  
Od lewej: oryginał · Standard syntetyczny · OpenVINO MiDaS v2.1 Small · OpenVINO DPT Large · Ensemble (fuzja wszystkich metod)*

---

> [!IMPORTANT]
> **Wagi modeli AI nie są przechowywane w repozytorium** (pliki przekraczają limit 100 MB narzucony przez GitHub).  
> Po sklonowaniu pobierz je jednym poleceniem:
> ```bash
> python download_models.py
> ```
> Modele są dostępne jako załączniki do [najnowszego GitHub Release](https://github.com/GrzegorzOle/DepthForge/releases/latest) (łącznie ~685 MB).  
> Bez modeli program działa wyłącznie w trybie **syntetycznym** (bez estymacji głębi przez AI).

---

## Przeznaczenie

DepthForge analizuje płaski obraz 2D i rekonstruuje ukrytą w nim **informację o głębi (przestrzeni)** — szacując, które fragmenty sceny są bliskie, a które odległe.  
Wygenerowane mapy głębokości stanowią podstawę do tworzenia **dotykowych reprodukcji 3D**, które umożliwiają osobom niewidomym fizyczne poznanie dzieł sztuki i eksponatów muzealnych poprzez dotyk.

Proces przebiega dwuetapowo:

1. **DepthForge** — automatyczna ekstrakcja głębi z obrazu przy użyciu modeli AI (MiDaS, DPT) przyspieszonych przez Intel OpenVINO.  
2. **Przygotowanie do druku 3D** — dane o głębi trafiają do specjalisty, który przygotowuje je w odpowiedniej formie fizycznej umożliwiającej eksplorację dotykową.  
   Za ten etap odpowiada **Jakub Oleksy**, specjalista ds. analizy druku 3D:  
   [linkedin.com/in/jakub-oleksy-672668333/](https://www.linkedin.com/in/jakub-oleksy-672668333/)

> **Wtyczka GIMP** — w pełni funkcjonalna wtyczka dla GIMP 3.2.x znajduje się w katalogu `gimp_plugins/`.  
> Instalacja: `python gimp_plugins/install_plugin.py` — skrypt automatycznie wykrywa ścieżkę projektu  
> i zapisuje `depthforge_install.json` w katalogu plug-ins GIMP-a, dzięki czemu wtyczka zawsze znajduje  
> właściwe modele, niezależnie od miejsca sklonowania projektu.  
> Funkcje: tryb wizualny (CLAHE) · tryb taktylny (parametry v9: fill-holes + detail-overlay + multiscale)  
> · wyjście kolorowe (INFERNO) lub w skali szarości · eksport STL.

---

## Wymagania

- Python 3.10+
- NumPy
- OpenCV (opencv-contrib-python)
- OpenVINO
- SciPy
- numpy-stl

Opcjonalnie — potrzebne wyłącznie do samodzielnej konwersji modeli
(`convert.py`, DPT → ONNX). Pipeline nigdy ich nie importuje, a pakiety CUDA
zajmują ~2 GB:

```bash
pip install -e ".[convert]"   # PyTorch + transformers
```

## Instalacja

### Dla użytkowników GIMP-a — pakiet samodzielny (zalecane)

Jeśli chcesz tylko wtyczki do GIMP-a, **nie potrzebujesz** Pythona, środowiska
wirtualnego ani tego repozytorium. Wybierz jeden plik ze
[strony wydań](https://github.com/GrzegorzOle/DepthForge/releases/latest):

| Platforma | Plik do pobrania | Instalacja | Modele |
|---|---|---|---|
| Windows 10/11 64-bit | `DepthForge-0.1.5-windows-x86_64-setup.exe` | uruchom | w środku |
| Linux x86_64 (glibc 2.28+) | `DepthForge-0.1.5-x86_64.AppImage` | `chmod +x` i uruchom | w środku |
| Dowolna, bez modeli | `…-linux-x86_64.tar.gz` / `…-windows-x86_64.zip` | `./install.sh` / `install.bat` | pobierane przy instalacji |

Dwa pierwsze to pakiety **offline** — niosą modele OpenVINO (~686 MB) w środku,
więc instalacja w ogóle nie wymaga internetu. Archiwa są mniejsze, ale dociągają
modele w trakcie instalacji.

Każdy pakiet zawiera własnego CPythona 3.12 z zainstalowanymi numpy, OpenCV,
OpenVINO i SciPy, więc jest niezależny zarówno od Pythona systemowego, jak i od
Pythona wbudowanego w GIMP-a (który jest w innej wersji i nie rozwiązuje tych
zależności). Instalacja kopiuje wtyczkę do GIMP-a i wskazuje jej interpreter z
pakietu.

Dwie rzeczy warte wiedzenia:

- **Nie kasuj pliku po instalacji.** AppImage *jest* interpreterem, który wywołuje
  GIMP, a na Windowsie trzyma go katalog instalacyjny. AppImage można przenieść —
  wystarczy uruchomić go ponownie z nowego miejsca.
- **AppImage wymaga FUSE**, jak każdy AppImage. Na systemie bez FUSE użyj
  `.tar.gz`.

Pełna instrukcja dla użytkownika jest w środku każdego pakietu jako
`INSTALL_PL.md` / `INSTALL_EN.md`, a w repozytorium leży w
`packaging/bundle_files/`.

### Dla programistów — ze źródeł

```bash
# Sklonuj repozytorium
git clone https://github.com/GrzegorzOle/DepthForge.git
cd DepthForge

# Utwórz środowisko wirtualne
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Zainstaluj wymagane biblioteki
pip install -r requirements.txt

# Pobierz modele OpenVINO (DPT Large + MiDaS v2.1 Small)
python download_models.py
```

> **Uwaga:** Pliki wag modeli nie są przechowywane w repozytorium (przekraczają limit 100 MB narzucony przez GitHub).  
> Są dystrybuowane jako załączniki do [GitHub Release](https://github.com/GrzegorzOle/DepthForge/releases/latest).  
> Skrypt `download_models.py` pobiera je automatycznie.

### Ręczne pobieranie modeli (opcjonalne)

```bash
python download_models.py --model dpt      # tylko DPT Large
python download_models.py --model midas    # tylko MiDaS v2.1 Small
python download_models.py --release v0.1.0 # konkretne wydanie
```

### Budowanie pakietów samodzielnych

Oba pakiety składa się **z Linuksa** — windowsowy przez pobranie wheeli
`win_amd64` poleceniem `pip --platform`, a nie przez uruchomienie interpretera
Windows:

```bash
python packaging/build_bundle.py                 # obie platformy → dist/
python packaging/build_bundle.py --target linux
python packaging/build_bundle.py --with-models   # z modelami w środku (~686 MB)
```

Oba pakiety offline powstają na bazie tego katalogu roboczego, więc najpierw
trzeba go zbudować z `--with-models --no-archive`:

```bash
python packaging/build_appimage.py               # → dist/DepthForge-x.y.z-x86_64.AppImage
python packaging/build_installer.py              # → dist/…-windows-x86_64-setup.exe
python packaging/assets/make_icon.py             # przegenerowanie ikony (jest w repo)
```

`build_installer.py` uruchamia Inno Setup pod wine w kontenerze
`amake/innosetup`, więc wymaga podmana albo dockera — ale nie wine w systemie.

Build linuksowy weryfikuje sam siebie, przepuszczając prawdziwy pipeline
taktylny przez dołączony interpreter, a build AppImage dodatkowo sprawdza
zapisaną konfigurację wtyczki. Ani pakiet windowsowy, ani plik .exe nie dają się
uruchomić na Linuksie, więc z założenia pozostają nieprzetestowane — instalator
trzeba odpalić na prawdziwym Windowsie przed wydaniem.

---

## Użycie

### Pipeline wizualny — mapy głębokości + STL

```bash
python src/depth_pipeline.py --input data/Stanczyk.jpg \
    --output-dir output/stanczyk \
    --width-mm 200 --relief-mm 12
```

### Pipeline taktylny — zoptymalizowany do odczytu dotykiem (druk tyflograficzny)

```bash
python src/depth_pipeline.py \
    --input data/Indian_summer_-_Google_Art_Project.jpg \
    --output-dir output/indian_summer_tactile \
    --tactile \
    --tactile-multiscale \
    --tactile-fine-sigma 1.5 \
    --tactile-limb-sigma 3.0 \
    --detail-strength 0.05 \
    --detail-blur-sigma 2.5 \
    --fill-holes \
    --width-mm 200 --relief-mm 7 --mesh-px 200
```

### Przetwarzanie wsadowe

```bash
python src/depth_forge.py --batch --input-dir data/ --output-dir output/
```

### Benchmark (wszystkie metody + ensemble)

```bash
python benchmark.py
```

---

## Przegląd pipeline'u

```
Obraz wejściowy
    │
    ├─► Syntetyczna mapa głębokości (Standard)
    ├─► OpenVINO MiDaS v2.1 Small
    └─► OpenVINO DPT Large
            │
            ▼
    Fuzja ensemble ze skalowaniem skali (DPT×0.50 + MiDaS×0.35 + Standard×0.15)
    Filtr guided (self-guided, zachowujący krawędzie)
            │
            ├─[--fill-holes]──► fill_small_object_holes()
            │                   Wypełnia płaskie wnętrza małych obiektów
            │                   (zwierzęta, dalekie postacie) niewykrytych przez modele
            │
    ┌───────┴──────────────────────────────────────────────┐
    │  Tryb WIZUALNY (domyślny)   Tryb TAKTYLNY (--tactile)    │
    │                                                      │
    │  apply_detail_overlay()     [--detail-strength > 0]      │
    │  Nakłada mikroteksturę      apply_detail_overlay() PRZED │
    │  z luminancji obrazu        wygładzaniem — przywraca     │
    │                             kontury kończyn z cieni      │
    │                                                      │
    │  postprocess_depth()        prepare_for_touch() lub      │
    │  CLAHE + łagodny Gauss      prepare_for_touch_multiscale()│
    │                             Usuwa drobny szum tkaniny,   │
    │                             zachowuje kontury kończyn    │
    │                                                      │
    │                             [--tactile-levels > 1]       │
    │                             quantize_depth_foreground_aware()│
    │                             Asymetryczna kwantyzacja:    │
    │                             bg_levels dla nieba/ziemi,   │
    │                             fg_levels dla postaci        │
    │                                                      │
    │                             smooth_quantized_boundaries()│
    │                             Morfologiczne domknięcie/    │
    │                             otwarcie na maskach poziomów │
    │                             — eliminuje staircase noise  │
    └──────────────────────────────────────────────────────┘
            │
            ▼
    depth_to_stl()  →  binarny plik STL (watertight, gotowy dla Prusa Slicer)
```

---

## Tryb taktylny — kompletny opis parametrów

Tryb taktylny (`--tactile`) jest zaprojektowany dla wydruków 3D, które będą **odczytywane dotykiem**, zgodnie z muzealnymi standardami tyflograficznymi (RNIB, Museo del Prado).

### Wygładzanie

| Flaga | Domyślnie | Opis |
|---|---|---|
| `--tactile-median` | `5` | Rozmiar filtra medianowego [px] — usuwa outliery szpilkowe przed Gaussem |
| `--tactile-sigma` | `3.5` | σ Gaussa [px] dla jednoprzebiegowego wygładzania (gdy `--tactile-multiscale` wyłączone) |
| `--tactile-multiscale` | wył. | **Wygładzanie wieloskalowe** — osobne usuwanie drobnej tekstury (tkanina, trawa) i zachowanie konturów kończyn (nogi, ręce) |
| `--tactile-fine-sigma` | `1.5` | σ [px] dla usuwania drobnej tekstury (zalecane 1.2–1.5) |
| `--tactile-limb-sigma` | `3.0` | σ [px] definiujący skalę kończyn; filtr końcowy używa `limb_sigma × 0.5`, by nie zlać sąsiednich nóg (zalecane 2.5–3.5) |

### Nakładka mikrodetalu w trybie taktylnym

W trybie taktylnym `--detail-strength > 0` uruchamia `apply_detail_overlay()` **przed** wygładzaniem.  
Przywraca kontury kończyn (separacja nóg, kierunek ramienia) z luminancji obrazu — informację, którą DPT/MiDaS często gubią przy postaciach w ciężkich szatach.  
Następne wygładzanie usuwa ostre igły, zachowując szersze pasma cienia kodujące pozycje kończyn.

| Flaga | Domyślnie | Opis |
|---|---|---|
| `--detail-strength` | `0.15` | Amplituda nakładki (0 = wyłączona). W trybie taktylnym użyj `0.05–0.08` |
| `--detail-blur-sigma` | `1.2` | Dolnoprzepustowe odcięcie [px] dla ekstrakcji detalu. W trybie taktylnym użyj `2.5`, by wyodrębnić szerokie pasma cienia zamiast drobnych igieł |

### Kwantyzacja z uwzględnieniem pierwszego planu

| Flaga | Domyślnie | Opis |
|---|---|---|
| `--tactile-levels` | `0` | Włącz dyskretne poziomy wysokości (ustaw > 1). Suma = `--tactile-bg-levels` + `--tactile-fg-levels` |
| `--tactile-fg-threshold` | `40.0` | Percentyl podziału tło/pierwszy plan |
| `--tactile-bg-levels` | `2` | Dyskretne poziomy dla strefy tła (niebo, ziemia) |
| `--tactile-fg-levels` | `4` | Dyskretne poziomy dla strefy pierwszego planu / postaci |
| `--tactile-boundary-kernel` | `9` | Rozmiar jądra morfologicznego [px] do wygładzania granic. Ustaw `0`, by wyłączyć |

### Wypełnianie wnętrz małych obiektów

| Flaga | Domyślnie | Opis |
|---|---|---|
| `--fill-holes` | wył. | Włącz po fuzji; wypełnia płaskie wnętrza małych obiektów (zwierzęta, dalekie postacie) |
| `--fill-holes-min-area` | `20` | Minimalna powierzchnia konturu [px²] |
| `--fill-holes-max-area` | `2000` | Maksymalna powierzchnia konturu [px²] — dopasuj do przybliżonej powierzchni pikselowej obiektu |
| `--fill-holes-kernel` | `5` | Jądro morfologiczne do zamykania konturów |

### Parametry STL

| Flaga | Domyślnie | Opis |
|---|---|---|
| `--width-mm` | `200` | Fizyczna szerokość modelu [mm] |
| `--relief-mm` | `10` (`7` z `--tactile`) | Maksymalna wysokość reliefu ponad płytą bazową [mm] |
| `--base-mm` | `3` | Grubość płyty bazowej [mm] |
| `--mesh-px` | `512` (`140` z `--tactile`) | Maksymalna rozdzielczość siatki STL [px]. Użyj 200–256 dla taktylnego |

---

## Zalecane presety taktylne

### Ciągły gradient (najlepszy punkt startowy)

```bash
python src/depth_pipeline.py --input obraz.jpg --output-dir output/tactile \
    --tactile --tactile-multiscale \
    --tactile-fine-sigma 1.5 --tactile-limb-sigma 3.0 \
    --detail-strength 0.05 --detail-blur-sigma 2.5 \
    --fill-holes \
    --width-mm 200 --relief-mm 7 --mesh-px 200
```

### Muzealny relief stopniowany (6 dyskretnych poziomów, fg-aware)

```bash
python src/depth_pipeline.py --input obraz.jpg --output-dir output/tactile_stepped \
    --tactile --tactile-multiscale \
    --tactile-fine-sigma 1.5 --tactile-limb-sigma 3.0 \
    --tactile-levels 6 --tactile-bg-levels 2 --tactile-fg-levels 4 \
    --tactile-fg-threshold 40 --tactile-boundary-kernel 9 \
    --width-mm 200 --relief-mm 7 --mesh-px 200
```

---

## Struktura projektu

```
DepthForge/
├── assets/                  # Zasoby statyczne (obrazy podglądowe itp.)
├── config.json              # Konfiguracja projektu
├── requirements.txt         # Wymagane biblioteki
├── benchmark.py             # Benchmark — wszystkie metody + ensemble
├── src/
│   ├── depth_forge.py       # Główny moduł generowania map głębokości (klasa DepthForge)
│   ├── depth_pipeline.py    # Pełny pipeline: głębokość → ensemble → taktylny → STL
│   │     Kluczowe funkcje:
│   │       normalize_f32_robust()             normalizacja percentylowa
│   │       fuse_depth_maps()                  fuzja ensemble ze skalowaniem skali
│   │       apply_detail_overlay()             mikrodetal z luminancji obrazu
│   │       fill_small_object_holes()          wypełnianie wnętrz małych obiektów
│   │       prepare_for_touch()                jednoprzebiegowe wygładzanie taktylne
│   │       prepare_for_touch_multiscale()     wieloskalowe wygładzanie taktylne
│   │       quantize_depth()                   kwantyzacja equal-area
│   │       quantize_depth_foreground_aware()  asymetryczna kwantyzacja tło/plan
│   │       smooth_quantized_boundaries()      morfologiczne wygładzanie granic poziomów
│   │       depth_to_stl()                     eksport watertight STL
│   │       run_pipeline()                     orkiestracja pełnego pipeline'u
│   │       run_pipeline_tactile()             wrapper z domyślnymi ustawieniami taktylnymi
├── gimp_plugins/
│   ├── depthforge/          # Wtyczka GIMP 3.x (folder ładowany przez GIMP)
│   └── install_plugin.py    # Skrypt instalacyjny wtyczki
├── data/                    # Obrazy wejściowe
├── models/
│   ├── midas/openvino/      # MiDaS v2.1 Small (OpenVINO IR)
│   └── dpt/openvino/        # DPT Large (OpenVINO IR)
└── output/                  # Wygenerowane mapy głębokości i pliki STL
```

---

## Konfiguracja

`config.json` kontroluje ścieżki do modeli i podstawowe ustawienia przetwarzania:

```json
{
  "model": {
    "depth_estimation": {
      "midas_model_path": "models/midas/openvino/midas_v21_small_256.xml",
      "dpt_model_path":   "models/dpt/openvino/dpt_large.xml"
    }
  }
}
```

---

## Uwagi projektowe — pipeline taktylny

Pipeline taktylny oparty jest na muzealnych wytycznych tyflograficznych (RNIB, Museo del Prado):

- **3–5 wyraźnie odróżnialnych poziomów wysokości** jest preferowanych nad ciągłym gradientem dla odczytu opuszkami palców
- **Staircase noise** na granicach poziomów jest eliminowany przez morfologiczne domknięcie/otwarcie na maskach **indeksów całkowitych** — nie wartości float, które przy błędach zaokrąglenia tworzą setki mikro-obszarów zamiast kilku czystych stref
- **Wygładzanie wieloskalowe** rozdziela drobny szum tekstury (~1–2 px, fałdy tkaniny, źdźbła trawy) od sensownej geometrii kończyn (~10–30 px, separacja nóg, kontury rąk) — bez jednego bliskozasięgowego Gaussa, który niszczyłby oba elementy jednocześnie
- **Kwantyzacja z uwzględnieniem pierwszego planu** zapobiega sytuacji, w której rozległe tło (niebo, ziemia) pochłania większość dostępnych poziomów kosztem głównej postaci — tło dostaje 2 poziomy, postać 4
- **Nakładka detalu przed wygładzaniem** przywraca informację o konturach kończyn z luminancji obrazu (którą DPT/MiDaS gubią przy postaciach w ciężkich szatach), a następne wygładzanie usuwa ostre igły, zachowując szersze pasma cienia kodujące pozycje nóg i rąk
