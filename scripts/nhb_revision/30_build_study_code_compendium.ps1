$ErrorActionPreference = "Stop"

$Root = "C:\Users\Gebruiker\Documents\TinyRNN State and Capacity"
$OutputDir = Join-Path $Root "state_capacity_tinyrnn\outputs\nhb_revision\record_package"
$OutputPath = Join-Path $OutputDir "state_capacity_complete_study_code_source.md"
$ManifestPath = Join-Path $OutputDir "state_capacity_complete_study_code_manifest.csv"
$GeneratedAt = (Get-Date).ToString("s")

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$IncludeRoots = @(
    "00_project_setup",
    "01_data_access_inventory",
    "02_behavioral_preprocessing",
    "03_simulation_recovery",
    "04_tiny_rnn_model",
    "05_state_capacity_estimation",
    "06_discovery_ds007554",
    "07_external_sleepybrain",
    "08_external_maus_workload",
    "09_dynamics_analysis",
    "10_baselines_robustness",
    "11_statistics_figures",
    "12_manuscript_outputs",
    "state_capacity_tinyrnn\scripts",
    "state_capacity_tinyrnn\src",
    "state_capacity_tinyrnn\tests"
)

$Files = New-Object System.Collections.Generic.List[System.IO.FileInfo]
foreach ($RelativeRoot in $IncludeRoots) {
    $FullRoot = Join-Path $Root $RelativeRoot
    if (Test-Path -LiteralPath $FullRoot) {
        Get-ChildItem -LiteralPath $FullRoot -Recurse -File |
            Where-Object {
                $_.Extension.ToLowerInvariant() -in @(".py", ".mjs", ".r", ".ps1") -and
                $_.FullName -notmatch "\\.venv\\|__pycache__|node_modules|\\.pytest_cache"
            } |
            ForEach-Object { $Files.Add($_) }
    }
}

$ReferencePackage = Join-Path $Root "state_capacity_tinyrnn\outputs\nhb_revision\record_package\state_capacity_method_analysis_reference.py"
if (Test-Path -LiteralPath $ReferencePackage) {
    $Files.Add((Get-Item -LiteralPath $ReferencePackage))
}

$UniqueFiles = $Files |
    Sort-Object FullName -Unique |
    Sort-Object {
        $_.FullName.Substring($Root.Length).TrimStart("\")
    }

$ManifestRows = foreach ($File in $UniqueFiles) {
    $RelativePath = $File.FullName.Substring($Root.Length).TrimStart("\")
    [PSCustomObject]@{
        relative_path = $RelativePath
        extension = $File.Extension
        bytes = $File.Length
        last_write_time = $File.LastWriteTime.ToString("s")
    }
}
$ManifestRows | Export-Csv -LiteralPath $ManifestPath -NoTypeInformation -Encoding UTF8

$Writer = [System.IO.StreamWriter]::new($OutputPath, $false, [System.Text.UTF8Encoding]::new($false))
try {
    $Writer.WriteLine("# Complete Study Code Source")
    $Writer.WriteLine("")
    $Writer.WriteLine("Generated: $GeneratedAt")
    $Writer.WriteLine("")
    $Writer.WriteLine("Workspace root: $Root")
    $Writer.WriteLine("")
    $Writer.WriteLine("This is a one-file source-code compendium for the TinyRNN State and Capacity study. It contains project-authored scripts, package source files, tests, and the generated method-analysis reference file. Downloaded raw-data utilities, virtual environments, caches, compiled files and dependency folders are intentionally excluded.")
    $Writer.WriteLine("")
    $Writer.WriteLine("## Manifest")
    $Writer.WriteLine("")
    $Writer.WriteLine("| # | Path | Language | Bytes |")
    $Writer.WriteLine("|---:|---|---|---:|")
    $Index = 1
    foreach ($File in $UniqueFiles) {
        $RelativePath = $File.FullName.Substring($Root.Length).TrimStart("\")
        $Language = switch ($File.Extension.ToLowerInvariant()) {
            ".py" { "python" }
            ".mjs" { "javascript" }
            ".r" { "r" }
            default { "text" }
        }
        $Writer.WriteLine("| $Index | $RelativePath | $Language | $($File.Length) |")
        $Index += 1
    }
    $Writer.WriteLine("")
    $Writer.WriteLine("## Source Files")
    $Writer.WriteLine("")

    foreach ($File in $UniqueFiles) {
        $RelativePath = $File.FullName.Substring($Root.Length).TrimStart("\")
        $Language = switch ($File.Extension.ToLowerInvariant()) {
            ".py" { "python" }
            ".mjs" { "javascript" }
            ".r" { "r" }
            default { "text" }
        }
        $Writer.WriteLine("")
        $Writer.WriteLine("---")
        $Writer.WriteLine("")
        $Writer.WriteLine("### $RelativePath")
        $Writer.WriteLine("")
        $Writer.WriteLine('```' + $Language)
        $Content = Get-Content -LiteralPath $File.FullName -Raw -Encoding UTF8
        $Writer.Write($Content)
        if (-not $Content.EndsWith("`n")) {
            $Writer.WriteLine("")
        }
        $Writer.WriteLine('```')
    }
}
finally {
    $Writer.Close()
}

Write-Output "Wrote $OutputPath"
Write-Output "Wrote $ManifestPath"
Write-Output "Files included: $($UniqueFiles.Count)"
