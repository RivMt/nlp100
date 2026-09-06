[CmdletBinding()]
param(
    [string]$OutDirectory = (Join-Path $PSScriptRoot 'out')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $OutDirectory -PathType Container)) {
    throw "Output directory not found: $OutDirectory"
}

$outPath = (Resolve-Path -LiteralPath $OutDirectory).Path
$problemDirectories = @(
    Get-ChildItem -LiteralPath $outPath -Directory -Force |
        Where-Object { $_.Name -match '^\d+$' } |
        ForEach-Object {
            $problemNumber = [int]$_.Name
            [pscustomobject]@{
                Directory = $_
                Chapter   = [math]::Floor($problemNumber / 10) + 1
                Problem   = $problemNumber
            }
        }
)

if ($problemDirectories.Count -eq 0) {
    throw "No numeric problem directories were found in: $outPath"
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$groups = $problemDirectories |
    Group-Object -Property Chapter |
    Sort-Object { [int]$_.Name }

foreach ($group in $groups) {
    $chapterName = 'chap{0:D2}' -f [int]$group.Name
    $archivePath = Join-Path $outPath "$chapterName.zip"
    $temporaryPath = Join-Path $outPath ('.{0}.{1}.tmp.zip' -f $chapterName, [guid]::NewGuid().ToString('N'))

    try {
        $archive = [System.IO.Compression.ZipFile]::Open(
            $temporaryPath,
            [System.IO.Compression.ZipArchiveMode]::Create
        )

        try {
            foreach ($problem in ($group.Group | Sort-Object Problem)) {
                $problemPath = $problem.Directory.FullName
                $files = @(Get-ChildItem -LiteralPath $problemPath -Recurse -File -Force)

                if ($files.Count -eq 0) {
                    [void]$archive.CreateEntry("$($problem.Directory.Name)/")
                    continue
                }

                foreach ($file in $files) {
                    $relativePath = $file.FullName.Substring($problemPath.Length).TrimStart('\', '/')
                    $entryName = ($problem.Directory.Name + '/' + $relativePath).Replace('\', '/')
                    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                        $archive,
                        $file.FullName,
                        $entryName,
                        [System.IO.Compression.CompressionLevel]::Optimal
                    ) | Out-Null
                }
            }
        }
        finally {
            if ($null -ne $archive) {
                $archive.Dispose()
            }
        }

        Move-Item -LiteralPath $temporaryPath -Destination $archivePath -Force
        Write-Host ("Created {0} ({1} problem directories)" -f $archivePath, $group.Count)
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}
