# One-key screen capture for thesis evidence images.
# ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 scripts as GBK
# and silently fails on Chinese literals.
#
# Usage (in PowerShell):
#   powershell -ExecutionPolicy Bypass -File E:\PetRecognition\capture.ps1
#   powershell -ExecutionPolicy Bypass -File E:\PetRecognition\capture.ps1 -Tag train
#   powershell -ExecutionPolicy Bypass -File E:\PetRecognition\capture.ps1 -Region 0,0,1600,900
#
# Output: E:\PetRecognition\figures\evidence\shot_<tag>_<yyyymmdd_HHMMSS>.png

param(
    [string]$Tag = "screen",
    [string]$Region = "",
    [string]$OutDir = "E:\PetRecognition\figures\evidence"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$file = Join-Path $OutDir ("shot_" + $Tag + "_" + $stamp + ".png")

$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
if ($Region -ne "") {
    $p = $Region.Split(",")
    $bounds = New-Object System.Drawing.Rectangle ([int]$p[0]),([int]$p[1]),([int]$p[2]),([int]$p[3])
}

$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$bmp.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()

Write-Output ("SAVED " + $file + "  " + $bounds.Width + "x" + $bounds.Height)
