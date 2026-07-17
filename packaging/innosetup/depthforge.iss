; DepthForge - Inno Setup script
; ==============================
; Wraps the offline Windows bundle (private CPython + deps + OpenVINO models +
; GIMP plugin) into a single setup .exe.
;
; The installer does not reinvent the wiring: it lays the bundle down as a normal
; directory and then runs the bundle's own bundle_install.py with the bundled
; interpreter. That script finds GIMP, copies the plugin and writes
; depthforge_install.json pointing at {app}\python\python.exe. Keeping that
; indirection is deliberate - the plugin is installed exactly one way on every
; platform and packaging format.
;
; Built from Linux via podman (see packaging/build_installer.py); ISPP is not
; used so the script stays compilable by a plain iscc.
;
; Expects these to be defined on the iscc command line (-D):
;   AppVersion, SourceDir, OutputDir, OutputBaseName

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\staging\DepthForge-1.6.0-windows-x86_64-offline"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif
#ifndef OutputBaseName
  #define OutputBaseName "DepthForge-setup"
#endif

[Setup]
AppId={{7C4E2F1A-9B3D-4E85-A6C7-DF1E2B3A4C50}
AppName=DepthForge
AppVersion={#AppVersion}
AppVerName=DepthForge {#AppVersion}
AppPublisher=DepthForge
AppSupportURL=https://github.com/Cdest-eu/DepthForge
DefaultDirName={autopf}\DepthForge
DefaultGroupName=DepthForge
DisableProgramGroupPage=yes
LicenseFile={#SourceDir}\app\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseName}
SetupIconFile=..\assets\depthforge.ico
UninstallDisplayIcon={app}\depthforge.ico
WizardStyle=modern

; Per-user install: no admin prompt, and GIMP's plug-ins directory is per-user
; anyway (%APPDATA%\GIMP), so nothing here needs elevation.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; The payload is ~1.2 GB, most of it OpenVINO float32 weights that compress
; poorly. lzma2/max buys little on those and costs a lot of build time and RAM;
; solid compression would additionally force a sequential extract of the whole
; archive. normal + non-solid keeps setup responsive and the build sane.
Compression=lzma2/normal
SolidCompression=no
DiskSpanning=no

[Languages]
Name: "pl"; MessagesFile: "compiler:Languages\Polish.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
pl.InstallingPlugin=Instalowanie wtyczki GIMP-a i sprawdzanie zaleznosci...
en.InstallingPlugin=Installing the GIMP plugin and verifying dependencies...
pl.PluginFailed=Pliki zostaly zainstalowane, ale rejestracja wtyczki w GIMP-ie nie powiodla sie.%n%nMozesz sprobowac ponownie skrotem "Zainstaluj wtyczke GIMP-a" w menu Start (przydatne, jesli GIMP-a instalujesz pozniej).%n%nSzczegoly:%n%1
en.PluginFailed=The files were installed, but registering the plugin with GIMP failed.%n%nYou can retry with the "Install GIMP plugin" shortcut in the Start menu (useful if you install GIMP later).%n%nDetails:%n%1
pl.RunInstaller=Zainstaluj wtyczke GIMP-a (uruchom ponownie po instalacji GIMP-a)
en.RunInstaller=Install GIMP plugin (re-run after installing GIMP)
pl.ViewReadme=Przeczytaj instrukcje instalacji
en.ViewReadme=Read the installation guide

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\assets\depthforge.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{cm:RunInstaller}"; Filename: "{app}\python\python.exe"; \
    Parameters: """{app}\bundle_install.py"""; WorkingDir: "{app}"; \
    IconFilename: "{app}\depthforge.ico"
Name: "{group}\{cm:ViewReadme}"; Filename: "{app}\INSTALL_PL.md"
Name: "{group}\{cm:UninstallProgram,DepthForge}"; Filename: "{uninstallexe}"

[UninstallRun]
; Runs before the files are removed, so the interpreter is still there. Takes the
; GIMP plugin and depthforge_install.json with it.
Filename: "{app}\python\python.exe"; Parameters: """{app}\bundle_install.py"" --uninstall"; \
    WorkingDir: "{app}"; Flags: runhidden waituntilterminated; RunOnceId: "DepthForgePluginUninstall"

[UninstallDelete]
; DepthForge() mkdir()s the config.json directories on construction, so app\ grows
; empty data/ and output/ dirs that Inno never tracked.
Type: filesandordirs; Name: "{app}\app\output"
Type: filesandordirs; Name: "{app}\app\data"
Type: filesandordirs; Name: "{app}\app\__pycache__"
Type: dirifempty; Name: "{app}"

[Code]
// Registering the plugin is done by the bundle's own installer rather than by
// [Run], so its exit code and output can actually be surfaced: it is the step
// most likely to fail on a user's machine (no GIMP yet, or an unexpected
// version), and a silent failure would leave a working install with an invisible
// plugin.
function RunPluginInstaller(var Output: String): Boolean;
var
  ResultCode: Integer;
  LogPath: String;
  Lines: TArrayOfString;
  I: Integer;
begin
  LogPath := ExpandConstant('{tmp}\depthforge_install_log.txt');

  // cmd /c so stdout can be redirected to a file we can read back.
  Result := Exec(
    ExpandConstant('{cmd}'),
    '/c ""' + ExpandConstant('{app}\python\python.exe') + '" "' +
      ExpandConstant('{app}\bundle_install.py') + '" > "' + LogPath + '" 2>&1"',
    ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);

  Output := '';
  if LoadStringsFromFile(LogPath, Lines) then
    for I := 0 to GetArrayLength(Lines) - 1 do
      Output := Output + Lines[I] + #13#10;

  Result := Result and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Output: String;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := ExpandConstant('{cm:InstallingPlugin}');
    if not RunPluginInstaller(Output) then
      MsgBox(FmtMessage(ExpandConstant('{cm:PluginFailed}'), [Output]),
             mbError, MB_OK);
  end;
end;
