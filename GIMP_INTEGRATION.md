# DepthForge – GIMP 3.x Plugin Integration

> **Wymagana wersja:** GIMP **3.2.x** lub nowszy  
> Wtyczka korzysta z nowego API GObject-Introspection (`gi.repository`) i **nie**
> jest kompatybilna z GIMP 2.x (stary Python-Fu / `gimpfu.register`).

---

## Struktura plików

```
gimp_plugins/
├── depthforge/               ← folder wymagany przez GIMP 3.x
│   └── depthforge.py         ← główny plik wtyczki
└── install_plugin.py         ← skrypt instalacyjny
```

GIMP 3.x wymaga, aby każda wtyczka Python znajdowała się we **własnym folderze**
o tej samej nazwie co plik `.py`.

---

## Instalacja automatyczna (zalecana)

```powershell
# Windows PowerShell – z katalogu projektu
python gimp_plugins\install_plugin.py

# Instalacja z jednoczesną instalacją zależności Python
python gimp_plugins\install_plugin.py --install-deps

# Odinstalowanie
python gimp_plugins\install_plugin.py --uninstall
```

Skrypt skopiuje folder `depthforge/` do:

| System  | Ścieżka                                                             |
|---------|---------------------------------------------------------------------|
| Windows | `%APPDATA%\GIMP\3.0\plug-ins\depthforge\`                           |
| Linux   | `~/.config/GIMP/3.0/plug-ins/depthforge/`                          |
| macOS   | `~/Library/Application Support/GIMP/3.0/plug-ins/depthforge/`      |

---

## Instalacja ręczna

1. Skopiuj folder `gimp_plugins/depthforge/` do katalogu plug-ins GIMP.
2. Na **Linux/macOS** nadaj uprawnienia wykonania:
   ```bash
   chmod +x ~/.config/GIMP/3.0/plug-ins/depthforge/depthforge.py
   ```
3. **Uruchom ponownie GIMP.**

---

## Wymagania Python

```powershell
pip install numpy opencv-python
```

Biblioteki `gi` (GObject Introspection) są dostarczane razem z GIMP 3.x –
**nie** wymagają osobnej instalacji pip.

---

## Użycie w GIMP

1. Otwórz dowolny obraz w GIMP (obsługiwane tryby: RGB i GRAY).
2. Przejdź do **Filtry → DepthForge → Generate Depth Map…**
3. Ustaw parametry w oknie dialogowym:

| Parametr | Opis |
|---|---|
| **Enhancement (0–100)** | Siła wzmocnienia kontrastu (CLAHE). 0 = brak, 100 = maksymalne. |
| **Invert depth** | Odwraca mapę głębi – jasne piksele = daleko, ciemne = blisko. |
| **Add as layer** | ✓ = dodaje wynik jako nową warstwę; odznacz, by otworzyć jako nowy obraz. |

4. Kliknij **OK** – mapa głębi pojawi się jako nowa warstwa **Depth Map (DepthForge)**.

### Relief warstwowy i maski planów (druk dotykowy)

Opcja **Relief warstwowy (plany)** odejmuje grunt i podnosi każdy obiekt jako
osobną płaszczyznę z wyczuwalną krawędzią: obiekt wyrasta z gruntu łagodnym
zboczem szerokości ok. 5 mm i ma zaokrąglony wierzch. Ma pierwszeństwo przed trybem
taktylnym v9. Eksport STL w tym trybie: relief 7 mm, siatka 400 px.

Tło dostaje faktury do rozróżniania palcem: szorstką trawę (mocniejszą na
pierwszym planie), wypukłe linie patyków i źdźbeł oraz łagodne chmury na
gładkim niebie. Obiekty i brzeg obrazu zostają bez faktury.

Bez dodatkowych kroków obiekty są wykrywane automatycznie. Żeby wskazać plany
ręcznie (pewniejsze, zwłaszcza we wnętrzach):

1. Zaznacz obiekt dowolnym narzędziem zaznaczania.
2. **Zaznaczenie → Zapisz do kanału**.
3. W oknie **Kanały** zmień nazwę kanału na `plan 1` (pierwszy plan – najwyżej),
   `plan 2` (drugi plan) itd. Kilka kanałów może mieć ten sam numer;
   dopisek po numerze jest dozwolony (`plan 2 pies`).
4. Uruchom DepthForge z zaznaczonym reliefem warstwowym. Wszystko poza
   kanałami to tło.

Kanały o innych nazwach są ignorowane. Kanał nazwany `plan …`, którego nie da
się odczytać, przerywa działanie z komunikatem zamiast po cichu przejść na
wykrywanie automatyczne.

---

## Jak działa generowanie mapy głębi

1. Piksele aktywnej warstwy są odczytywane przez **bufor GEGL** (natywne API GIMP 3.x).
2. Jeśli w katalogu `src/` projektu dostępny jest pipeline DepthForge
   (`depth_forge.py` + modele OpenVINO), wtyczka korzysta z modelu MiDaS.
3. Gdy modele są niedostępne, stosowany jest **fallback syntetyczny**:
   odwrócenie luminancji + mapa krawędzi (Laplacian) + CLAHE.
4. Wynik jest wstawiany z powrotem do GIMP przez **shadow buffer GEGL**.

---

## Wywoływanie ze Script-Fu Console / Batch

```scheme
(plug-in-depthforge RUN-NONINTERACTIVE image drawable 75 FALSE TRUE)
;; argumenty: enhancement-level  invert-depth  add-as-layer
```

---

## Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---|---|
| Brak menu „Filters → DepthForge" | Sprawdź, czy folder `depthforge/` jest we właściwym katalogu plug-ins; uruchom GIMP ponownie. |
| „Missing Python dependency" | `pip install numpy opencv-python` w środowisku Python GIMP. |
| Błąd uprawnień (Linux/macOS) | `chmod +x ~/.config/GIMP/3.0/plug-ins/depthforge/depthforge.py` |
| Mapa głębi jest szara/płaska | Zainstaluj modele OpenVINO (patrz `MODELS.md`) lub zwiększ Enhancement do 80+. |

---

## Zgodność

| GIMP    | Status                              |
|---------|-------------------------------------|
| 3.2.x   | ✅ Obsługiwane (GIMP 3.2.4)         |
| 3.0.x   | ✅ Powinno działać                   |
| 2.10.x  | ❌ Nie obsługiwane (stare gimpfu API)|
