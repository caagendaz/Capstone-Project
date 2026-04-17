$target = 1000
$project = 'C:\Users\cagns\Downloads\capstone_project'
$hitsDir = Join-Path $project 'data\external\amrfinder_batch_cipro_remaining\hits'
$logPath = Join-Path $project 'results\NCBI\stop_at_1000.log'
while ($true) {
    $count = @(Get-ChildItem $hitsDir -Filter *.tsv -ErrorAction SilentlyContinue).Count
    "$(Get-Date -Format s) hits=$count" | Out-File -FilePath $logPath -Append -Encoding utf8
    if ($count -ge $target) {
        $procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python.exe' -and $_.CommandLine -match 'process_remaining_ncbi_ciprofloxacin.py' }
        foreach ($p in $procs) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            "$(Get-Date -Format s) stopped_python_pid=$($p.ProcessId)" | Out-File -FilePath $logPath -Append -Encoding utf8
        }
        wsl --shutdown
        "$(Get-Date -Format s) wsl_shutdown" | Out-File -FilePath $logPath -Append -Encoding utf8
        break
    }
    Start-Sleep -Seconds 60
}
