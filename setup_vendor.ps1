$ErrorActionPreference = "Stop"

$Version = "0.10.35"
$PackageBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@$Version"
$ModelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VendorRoot = Join-Path $Root "vendor\mediapipe"
$WasmRoot = Join-Path $VendorRoot "wasm"
$ModelRoot = Join-Path $Root "models"

New-Item -ItemType Directory -Force -Path $WasmRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null

$Files = @(
    "vision_bundle.mjs",
    "wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_module_internal.js",
    "wasm/vision_wasm_module_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm"
)

foreach ($File in $Files) {
    $RelativeWindowsPath = $File.Replace("/", "\")
    $Destination = Join-Path $VendorRoot $RelativeWindowsPath
    $DestinationDirectory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null

    Write-Host "Download: $File"
    Invoke-WebRequest -Uri "$PackageBase/$File" -OutFile $Destination
}

Write-Host "Download: hand_landmarker.task"
Invoke-WebRequest -Uri $ModelUrl -OutFile (Join-Path $ModelRoot "hand_landmarker.task")

Write-Host ""
Write-Host "準備完了です。vendor/ と models/ を含めてGitHubへアップロードしてください。"
