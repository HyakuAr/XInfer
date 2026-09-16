$cmd = 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul 2>&1 && build\verify_pcie_and_memory.exe out\qwen3_8_27b.xinfer'
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$cmd`"" -PassThru

# Wait 10 seconds for model loading into VRAM
Start-Sleep -Seconds 10


Write-Host "Sampling GPU Memory & Engines during active decode loop..."
for ($i = 0; $i -lt 5; $i++) {
    $samples = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage', '\GPU Engine(*copy*)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples
    
    $dedicatedStr = ""
    foreach ($s in ($samples | Where-Object { $_.Path -like "*dedicated*" })) {
        $valGB = [math]::Round($s.CookedValue / 1GB, 2)
        if ($valGB -gt 0.1) {
            $dedicatedStr += "$($s.InstanceName): $valGB GB | "
        }
    }

    $copyMax = 0.0
    foreach ($s in ($samples | Where-Object { $_.Path -like "*copy*" })) {
        if ($s.CookedValue -gt $copyMax) { $copyMax = $s.CookedValue }
    }

    Write-Host "Sample ${i}: Dedicated VRAM: $dedicatedStr Copy Engine (PCIe) Max Util = $([math]::Round($copyMax, 2)) %"
    Start-Sleep -Seconds 1
}

$proc.WaitForExit()
Write-Host "Process exited with code $($proc.ExitCode)"

