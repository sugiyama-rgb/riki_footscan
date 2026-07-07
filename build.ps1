Set-Location $PSScriptRoot
python -m PyInstaller --clean editor.spec
Write-Host "完了: dist\insoleEDIT.exe"
