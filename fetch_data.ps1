$ErrorActionPreference = "Continue"
$dst = "E:\PetRecognition\data\raw"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$log = "E:\PetRecognition\logs\fetch.log"
function W($m) { "$(Get-Date -Format HH:mm:ss) $m" | Out-File -Append -Encoding ascii $log }
W "START"

$jobs = @(
  @{ url = "https://hf-mirror.com/datasets/timm/oxford-iiit-pet/resolve/main/data/train-00000-of-00001.parquet"; out = "pet_train.parquet" },
  @{ url = "https://hf-mirror.com/datasets/timm/oxford-iiit-pet/resolve/main/data/test-00000-of-00001.parquet";  out = "pet_test.parquet"  },
  @{ url = "https://hf-mirror.com/datasets/amaye15/stanford-dogs/resolve/main/data/train-00000-of-00002.parquet"; out = "dogs_train1.parquet" },
  @{ url = "https://hf-mirror.com/datasets/amaye15/stanford-dogs/resolve/main/data/train-00001-of-00002.parquet"; out = "dogs_train2.parquet" },
  @{ url = "https://hf-mirror.com/datasets/amaye15/stanford-dogs/resolve/main/data/test-00000-of-00001.parquet";  out = "dogs_test.parquet"  }
)

foreach ($j in $jobs) {
  $p = Join-Path $dst $j.out
  if ((Test-Path $p) -and ((Get-Item $p).Length -gt 10000000)) { W "skip $($j.out) exists"; continue }
  W "get $($j.out)"
  & curl.exe -sSL --fail -C - --retry 8 --retry-delay 5 --retry-all-errors --max-time 3600 -o $p $j.url
  if ($LASTEXITCODE -eq 0 -and (Test-Path $p)) {
    $sz = [math]::Round((Get-Item $p).Length / 1MB, 1)
    W "done $($j.out) = $sz MB"
  } else {
    W "FAIL $($j.out) exit=$LASTEXITCODE"
  }
}
W "ALL_DONE"
