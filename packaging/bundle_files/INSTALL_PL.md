# DepthForge — instrukcja instalacji wtyczki do GIMP-a

*(English version: `INSTALL_EN.md`)*

DepthForge generuje mapy głębi z płaskich obrazów 2D (głównie obrazów muzealnych)
i eksportuje pliki STL do druku 3D — **reprodukcji tyflograficznych czytanych
opuszkami palców** przez osoby niewidome i słabowidzące.

Ten pakiet jest **całkowicie samodzielny**. Zawiera własnego Pythona wraz ze
wszystkimi bibliotekami. **Nie musisz mieć Pythona w systemie** i nie musisz
sprawdzać, jakiej wersji Pythona używa Twój GIMP — pakiet go w ogóle nie używa.

---

## 0. Który plik pobrałeś?

DepthForge jest wydawany w kilku postaciach. Znajdź swoją — dwie pierwsze robią
wszystko za Ciebie i na tej sekcji możesz skończyć czytanie.

| Plik | Co zrobić | Modele |
|---|---|---|
| `…-setup.exe` | Uruchom i przeklikaj kreatora. Sam zainstaluje wtyczkę. | w środku |
| `…-x86_64.AppImage` | Nadaj prawo wykonywania (`chmod +x`) i uruchom. Sam zainstaluje wtyczkę. | w środku |
| `…-windows-x86_64.zip` | Rozpakuj i wykonaj sekcję 2 poniżej. | pobierane przy instalacji |
| `…-linux-x86_64.tar.gz` | Rozpakuj i wykonaj sekcję 2 poniżej. | pobierane przy instalacji |

Niezależnie od wyboru: **zostaw plik lub folder tam, gdzie jest** po instalacji.
GIMP sięga po Pythona w jego środku przy każdym użyciu wtyczki. Przeniesienie
jest w porządku — wystarczy uruchomić instalator (albo AppImage) jeszcze raz z
nowego miejsca.

AppImage wymaga **FUSE**, jak każdy AppImage. Jeśli odmawia startu z komunikatem
o `libfuse` albo `fusermount`, doinstaluj pakiet FUSE 2 swojej dystrybucji albo
użyj `.tar.gz`.

---

## 1. Czego potrzebujesz

| | |
|---|---|
| **GIMP** | wersja 3.2 lub nowsza |
| **System** | Windows 10/11 64-bit **albo** Linux 64-bit (glibc 2.28+: Ubuntu 20.04+, Debian 10+, Fedora 29+) |
| **Miejsce na dysku** | ok. 1,5 GB (pakiet + modele) |
| **Internet** | tylko dla pakietów `.zip` / `.tar.gz`, które pobierają modele (ok. 686 MB) przy instalacji. `.exe` i `.AppImage` niosą modele w środku i instalują się całkowicie offline. |
| **Pamięć RAM** | minimum 4 GB, zalecane 8 GB |

Wtyczka liczy na procesorze (CPU). Karta graficzna nie jest potrzebna.

---

## 2. Instalacja krok po kroku

### Krok 1 — rozpakuj pakiet

Wypakuj archiwum w miejscu, gdzie ma **zostać na stałe**:

* **Windows** — kliknij plik `.zip` prawym przyciskiem → *Wyodrębnij wszystkie…*
  Dobre miejsce: `C:\DepthForge`
* **Linux** — `tar -xzf DepthForge-<wersja>-linux-x86_64.tar.gz`
  Dobre miejsce: `~/DepthForge` lub `/opt/DepthForge`

> ⚠️ **To ważne:** GIMP będzie sięgał do Pythona wewnątrz tego folderu.
> **Nie przenoś ani nie usuwaj tego folderu po instalacji.** Nie rozpakowuj go
> do katalogu *Pobrane*, jeśli masz zwyczaj go czyścić. Jeśli musisz przenieść
> folder — przenieś go, a potem uruchom instalator jeszcze raz z nowego miejsca.

### Krok 2 — zamknij GIMP-a

Jeśli GIMP jest uruchomiony, zamknij go teraz. Wtyczki są wczytywane przy starcie.

### Krok 3 — uruchom instalator

* **Windows** — kliknij dwukrotnie **`install.bat`**

  Jeśli pojawi się ostrzeżenie *„System Windows ochronił Twój komputer"*,
  kliknij **Więcej informacji → Uruchom mimo to**. To standardowy komunikat dla
  plików pobranych z internetu, które nie są podpisane cyfrowo.

* **Linux** — otwórz terminal w folderze z pakietem i wpisz:

  ```bash
  ./install.sh
  ```

Instalator sam:

1. znajdzie katalog wtyczek GIMP-a,
2. skopiuje tam wtyczkę,
3. zapisze ścieżkę do Pythona z pakietu,
4. pobierze modele (ok. 686 MB — to może potrwać kilka minut),
5. sprawdzi, czy wszystko działa.

Na końcu zobaczysz `=== Done ===`. Jeśli zamiast tego pojawi się `[FAIL]`,
zajrzyj do sekcji **Rozwiązywanie problemów** poniżej.

### Krok 4 — uruchom GIMP-a

Wtyczka znajduje się w menu:

```
Filtry  →  DepthForge  →  Generate Depth Map…
```

---

## 3. Pierwsze użycie

1. Otwórz w GIMP-ie obraz (**Plik → Otwórz**).
2. Wybierz **Filtry → DepthForge → Generate Depth Map…**
3. Wybierz tryb:
   * **Tryb wizualny** — ładna mapa głębi do oglądania na ekranie.
   * **Tryb tyflograficzny (dotykowy)** — do druku 3D czytanego dotykiem.
     Efekt jest celowo *odwrotnością* ładnej mapy: drobna faktura to szum,
     kontury muszą być szerokie i płytkie, a 3–5 wyraźnych poziomów wysokości
     czyta się palcem lepiej niż płynny gradient. Ten tryb realizuje wytyczne
     tyflograficzne (RNIB, Museo del Prado).
4. Pierwsze uruchomienie trwa dłużej (ładowanie modeli do pamięci).
   Obraz 2000×1500 px liczy się zwykle kilkadziesiąt sekund do kilku minut,
   zależnie od procesora.

> **Uwaga:** GIMP w trakcie liczenia wygląda, jakby się zawiesił — okno nie
> odpowiada, bo praca dzieje się w osobnym procesie. To normalne. Poczekaj.

---

## 4. Rozwiązywanie problemów

### Wtyczki nie ma w menu Filtry

* Czy GIMP został uruchomiony **ponownie** po instalacji?
* Sprawdź, czy Twój GIMP używa innego katalogu konfiguracji, i wskaż go ręcznie:

  ```bash
  # Linux
  ./install.sh --gimp-dir ~/.config/GIMP/3.2
  ```
  ```bat
  rem Windows (w wierszu poleceń, w folderze pakietu)
  install.bat --gimp-dir "%APPDATA%\GIMP\3.2"
  ```

  Ścieżkę do swojego katalogu znajdziesz w GIMP-ie:
  **Edycja → Preferencje → Katalogi → Wtyczki**.

### Wtyczka jest w menu, ale zgłasza błąd

Uruchom wbudowaną diagnostykę: **Filtry → DepthForge → Diagnose…**
Pokaże, czy wtyczka widzi Pythona z pakietu i katalog projektu.

Szczegółowy log znajdziesz w pliku `depthforge_gimp.log` w podfolderze `app/`
tego pakietu (albo w katalogu tymczasowym systemu).

### „Mapa głębi jest płaska i nieciekawa"

To prawie zawsze znaczy, że **nie ma modeli** — DepthForge cicho przełącza się
wtedy na zapasowy estymator syntetyczny (odwrócona jasność + rozmycie), który
daje słabe wyniki. Dociągnij modele:

```bash
# Linux
./python/bin/python3 app/download_models.py
```
```bat
rem Windows
python\python.exe app\download_models.py
```

### Przeniosłem folder i wtyczka przestała działać

Uruchom `install.sh` / `install.bat` jeszcze raz — z nowego miejsca. Instalator
zapisze zaktualizowane ścieżki.

### Pobieranie modeli nie powiodło się

Instalator można uruchomić ponownie w dowolnej chwili — pliki już pobrane nie
będą pobierane drugi raz. Jeśli sieć jest zablokowana, pobierz cztery pliki
ręcznie ze strony <https://github.com/GrzegorzOle/DepthForge/releases> i umieść
je w pakiecie:

```
app/models/dpt/openvino/dpt_large.bin
app/models/dpt/openvino/dpt_large.xml
app/models/midas/openvino/midas_v21_small_256.bin
app/models/midas/openvino/midas_v21_small_256.xml
```

---

## 5. Odinstalowanie

* **Windows** — kliknij dwukrotnie `uninstall.bat`
* **Linux** — `./uninstall.sh`

To usuwa wtyczkę z GIMP-a. Sam folder pakietu zostaje — skasuj go ręcznie,
żeby odzyskać miejsce na dysku.

---

## 6. Co jest w środku

```
DepthForge-<wersja>-<platforma>/
├── install.sh / install.bat        ← uruchamiasz to
├── uninstall.sh / uninstall.bat
├── python/                          Python 3.12 + numpy, OpenCV, OpenVINO, SciPy
├── app/                             kod DepthForge + modele (po instalacji)
├── plugin/depthforge/               wtyczka kopiowana do GIMP-a
├── INSTALL_PL.md                    ten plik
└── INSTALL_EN.md                    wersja angielska
```

Pakiet nie instaluje niczego w systemie poza jednym folderem wtyczki w katalogu
konfiguracyjnym GIMP-a. Nie zmienia rejestru, nie dotyka Twojego Pythona ani
Pythona GIMP-a.

Licencja: patrz `app/LICENSE`.
Kod źródłowy i zgłoszenia błędów: <https://github.com/GrzegorzOle/DepthForge>
