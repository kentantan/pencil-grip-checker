$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$RuntimeFiles = @(
    "index.html",
    "app.js",
    "db.js"
)

$ForbiddenPatterns = @(
    "https?://",
    "fetch\s*\(",
    "XMLHttpRequest",
    "WebSocket",
    "sendBeacon",
    "EventSource"
)

$Findings = @()
foreach ($RelativePath in $RuntimeFiles) {
    $Path = Join-Path $Root $RelativePath
    foreach ($Pattern in $ForbiddenPatterns) {
        $Matches = Select-String -Path $Path -Pattern $Pattern -AllMatches
        foreach ($Match in $Matches) {
            $Findings += "{0}:{1}: {2}" -f $RelativePath, $Match.LineNumber, $Match.Line.Trim()
        }
    }
}

$IndexPath = Join-Path $Root "index.html"
$CspOk = Select-String -Path $IndexPath -SimpleMatch "connect-src 'self'" -Quiet

if ($Findings.Count -gt 0) {
    Write-Host "外部通信につながり得る記述が見つかりました:" -ForegroundColor Red
    $Findings | ForEach-Object { Write-Host $_ }
    exit 1
}

if (-not $CspOk) {
    Write-Host "CSPの connect-src 'self' が見つかりません。" -ForegroundColor Red
    exit 1
}

Write-Host "確認OK: アプリ本体に外部URL・送信用APIは見つからず、CSPは同一生成元通信だけを許可しています。" -ForegroundColor Green
Write-Host "注: sw.jsのfetchは同一生成元ファイルのオフラインキャッシュ用途です。"
