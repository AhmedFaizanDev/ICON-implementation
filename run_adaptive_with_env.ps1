param(
    [string]$Config = "config.json",
    [string]$CsvFile = "data/200.csv",
    [int]$Start = 0,
    [int]$Count = 1,
    [string]$Output = "output/adaptive_first2_recent_config.jsonl"
)

$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        return
    }
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) {
            return
        }
        $name = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim().Trim('"')
        [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

Import-DotEnv -Path ".env"

$orKey = $env:OPENROUTER_API_KEY
if (-not $orKey) {
    throw "OPENROUTER_API_KEY is not set. Add it to .env or current shell env."
}

if (-not $env:OPENROUTER_API_BASE) {
    $env:OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
}

$env:ROUTER_LLM_API_KEY = $orKey
$env:PAPER_GENERATOR_API_KEY = $orKey
$env:SCRIPT_GENERATOR_API_KEY = $orKey
$env:CASE_STUDY_GENERATOR_API_KEY = $orKey
$env:CTI_BRIEFING_GENERATOR_API_KEY = $orKey
$env:RCA_REPORT_GENERATOR_API_KEY = $orKey
$env:TARGET_LLM_API_KEY = $orKey
$env:JUDGE_LLM_API_KEY = $orKey
$env:REFLECTOR_LLM_API_KEY = $orKey

$env:ROUTER_LLM_API_BASE = $env:OPENROUTER_API_BASE
$env:PAPER_GENERATOR_API_BASE = $env:OPENROUTER_API_BASE
$env:SCRIPT_GENERATOR_API_BASE = $env:OPENROUTER_API_BASE
$env:CASE_STUDY_GENERATOR_API_BASE = $env:OPENROUTER_API_BASE
$env:CTI_BRIEFING_GENERATOR_API_BASE = $env:OPENROUTER_API_BASE
$env:RCA_REPORT_GENERATOR_API_BASE = $env:OPENROUTER_API_BASE
$env:TARGET_LLM_API_BASE = $env:OPENROUTER_API_BASE
$env:JUDGE_LLM_API_BASE = $env:OPENROUTER_API_BASE
$env:REFLECTOR_LLM_API_BASE = $env:OPENROUTER_API_BASE

& ".\.venv\Scripts\python.exe" "run_adaptive.py" `
    --config $Config `
    --csv-file $CsvFile `
    --start $Start `
    --count $Count `
    --output $Output

