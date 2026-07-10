Set-Location $PSScriptRoot

$version = Get-Date -Format "yyyy.MM.dd"
$versionFile = "version.py"
$versionContent = Get-Content $versionFile -Raw -Encoding utf8
$versionContent = $versionContent -replace 'VERSION:\s*str\s*=\s*".*"', "VERSION: str = `"$version`""
Set-Content -Path $versionFile -Value $versionContent -NoNewline -Encoding utf8

try {
    python -m PyInstaller --clean editor.spec

    $builtExe = "dist\insoleEDIT.exe"
    if (-not (Test-Path $builtExe)) {
        throw "PyInstallerのビルドに失敗しました: $builtExe が見つかりません"
    }

    $candidateName = "insoleEDIT_$version.exe"
    $suffix = 2
    while (Test-Path "dist\$candidateName") {
        $candidateName = "insoleEDIT_${version}_$suffix.exe"
        $suffix++
    }

    Rename-Item -Path $builtExe -NewName $candidateName
    Write-Host "完了: dist\$candidateName"
}
finally {
    $versionContent = Get-Content $versionFile -Raw -Encoding utf8
    $versionContent = $versionContent -replace 'VERSION:\s*str\s*=\s*".*"', 'VERSION: str = "dev"'
    Set-Content -Path $versionFile -Value $versionContent -NoNewline -Encoding utf8
}
