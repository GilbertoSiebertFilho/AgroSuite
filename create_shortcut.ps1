# Puts an AgroSuite shortcut in the Start menu, and on the desktop when it
# can: run it once after copying the folder to a computer. The shortcut
# opens run.bat, with the app's icon. Moving the folder afterwards breaks
# the shortcut; run this again from the new place.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$shell = New-Object -ComObject WScript.Shell
$made = @()

foreach ($folder in @([Environment]::GetFolderPath('Programs'),
                      [Environment]::GetFolderPath('Desktop'))) {
    try {
        $link = $shell.CreateShortcut((Join-Path $folder 'AgroSuite.lnk'))
        $link.TargetPath = Join-Path $root 'run.bat'
        $link.WorkingDirectory = $root
        $link.IconLocation = (Join-Path $root 'AgroSuite.ico') + ',0'
        $link.Description = 'AgroSuite - open the app in the browser'
        $link.Save()
        $made += $folder
    } catch {
        # A desktop on a full network drive refuses even a shortcut; the
        # Start menu one is enough to find and pin the app.
        Write-Host "  Could not write to $folder"
        Write-Host "  ($($_.Exception.Message.Trim()))"
    }
}

if ($made.Count -eq 0) {
    Write-Host ""
    Write-Host "  No shortcut could be made. Open the app with run.bat instead."
    exit 1
}
Write-Host ""
Write-Host "  AgroSuite shortcut made in:"
foreach ($folder in $made) { Write-Host "    $folder" }
Write-Host ""
Write-Host "  Press the Windows key and type AgroSuite. Right-click it to pin it"
Write-Host "  to the taskbar."
