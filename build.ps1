$buildDir = "d:\SOLO-3\sn-neutron-94\build"
$srcDir = "d:\SOLO-3\sn-neutron-94"
$pyPkgDir = "d:\SOLO-3\sn-neutron-94\python\snsolver"

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
}

Push-Location $buildDir
cmake .. -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Error "CMake configuration failed!"
    exit 1
}

cmake --build . --config Release
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Error "Build failed!"
    exit 1
}
Pop-Location

$pydFile = Get-ChildItem -Path $buildDir -Filter "_snsolver*.pyd" -Recurse | Select-Object -First 1
if ($pydFile) {
    Copy-Item $pydFile.FullName $pyPkgDir -Force
    Write-Host "Copied $($pydFile.Name) to python/snsolver/"
} else {
    Write-Error "Could not find built .pyd file!"
    exit 1
}

Write-Host "Build and install complete!"
