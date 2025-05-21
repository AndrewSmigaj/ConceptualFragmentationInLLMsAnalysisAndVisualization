# safe_cleanup.ps1
# A safer approach to repository cleanup with list-only mode, selective cleanup, and verification

param (
    [switch]$ListOnly = $false,                     # Only list files that would be removed, don't delete
    [switch]$CleanBackups = $false,                 # Clean backup files (.bak, etc.)
    [switch]$CleanCache = $false,                   # Clean cache files
    [switch]$CleanTemp = $false,                    # Clean temporary files
    [switch]$CleanPycache = $false,                 # Clean Python cache (__pycache__, *.pyc)
    [switch]$CleanDuplicates = $false,              # Clean duplicate/outdated files
    [switch]$All = $false,                          # Clean everything
    [switch]$VerifyDashboard = $false,              # Verify dashboard works after cleanup
    [switch]$CreateBackup = $true                   # Create backup before cleanup
)

function Write-Title($text) {
    Write-Host "`n$text" -ForegroundColor Cyan
    Write-Host ("-" * $text.Length) -ForegroundColor Cyan
}

function Write-Section($text) {
    Write-Host "`n$text" -ForegroundColor Green
}

function Write-ListItem($text) {
    Write-Host "  - $text" -ForegroundColor White
}

function Write-Action($text, $target) {
    if ($ListOnly) {
        Write-Host "  Would remove: $target" -ForegroundColor Gray
    } else {
        Write-Host "  Removing: $target" -ForegroundColor Gray
    }
}

function Get-SafeFilePath($filePath) {
    # Convert relative paths to absolute if needed
    if (-not [System.IO.Path]::IsPathRooted($filePath)) {
        $filePath = Join-Path (Get-Location) $filePath
    }
    return $filePath
}

function Test-DashboardFunctionality {
    Write-Section "Testing dashboard functionality..."
    
    # Use the verification script instead of starting the actual server
    $verifyResult = Start-Process -FilePath "python" -ArgumentList "visualization/dash_verify.py" -Wait -NoNewWindow -PassThru
    
    # Check the exit code
    if ($verifyResult.ExitCode -eq 0) {
        Write-Host "  Dashboard verification successful!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  ERROR: Dashboard verification failed with exit code: $($verifyResult.ExitCode)" -ForegroundColor Red
        Write-Host "  The dashboard may be broken after cleanup. Consider restoring from backup." -ForegroundColor Red
        return $false
    }
}

# Display script header
Write-Title "Safe Repository Cleanup Script"
Write-Host "This script provides safe cleanup options with verification and backups." -ForegroundColor White

# If no specific cleanup options selected and not listing, show help
if (-not ($ListOnly -or $CleanBackups -or $CleanCache -or $CleanTemp -or $CleanPycache -or $CleanDuplicates -or $All)) {
    Write-Host "`nUsage: ./safe_cleanup.ps1 [-ListOnly] [-CleanBackups] [-CleanCache] [-CleanTemp] [-CleanPycache] [-CleanDuplicates] [-All] [-VerifyDashboard] [-CreateBackup]`n" -ForegroundColor Yellow
    Write-Host "Examples:"
    Write-Host "  ./safe_cleanup.ps1 -ListOnly -All                    # Just list all files that would be removed" -ForegroundColor Gray
    Write-Host "  ./safe_cleanup.ps1 -CleanBackups -CleanPycache       # Remove backup files and Python cache" -ForegroundColor Gray
    Write-Host "  ./safe_cleanup.ps1 -All -VerifyDashboard             # Clean everything and verify dashboard works" -ForegroundColor Gray
    exit
}

# Determine which cleanup operations to perform
$performCleanBackups = $CleanBackups -or $All
$performCleanCache = $CleanCache -or $All 
$performCleanTemp = $CleanTemp -or $All
$performCleanPycache = $CleanPycache -or $All
$performCleanDuplicates = $CleanDuplicates -or $All

if ($ListOnly) {
    Write-Host "`nList-only mode: Files will not actually be removed`n" -ForegroundColor Yellow
}

# Create stats collectors
$totalFilesIdentified = 0
$totalSizeIdentified = 0
$filesToRemove = @()

# 0. Create comprehensive backup if requested
if ($CreateBackup -and -not $ListOnly) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = "safe_backup_$timestamp"
    
    Write-Section "Creating comprehensive backup in $backupDir..."
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    # Critical files to back up
    $criticalFiles = @(
        "visualization/dash_app.py",
        "visualization/run_dashboard.py",
        "visualization/llm_tab.py",
        "visualization/data_interface.py",
        "visualization/reducers.py",
        "visualization/traj_plot.py",
        "concept_fragmentation/foundations.md",
        "run_dashboard.bat",
        "run_cluster_paths.py",
        "run_full_pipeline.ps1",
        "run_analysis.py"
    )
    
    foreach ($file in $criticalFiles) {
        if (Test-Path $file) {
            $targetDir = Join-Path $backupDir (Split-Path -Parent $file)
            if (-not (Test-Path $targetDir)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }
            Copy-Item $file -Destination (Join-Path $backupDir $file) -Force
            Write-Host "  Backed up $file" -ForegroundColor Gray
        }
    }
    
    Write-Host "  Created backup in $backupDir" -ForegroundColor Green
}

# 1. Identify backup files
if ($performCleanBackups) {
    Write-Section "Identifying backup files (.bak.*, .bak)..."
    
    # Get .bak.* files
    $bakExtFiles = Get-ChildItem -Path . -Recurse -File | Where-Object { $_.FullName -like "*.bak.*" }
    foreach ($file in $bakExtFiles) {
        $totalFilesIdentified++
        $totalSizeIdentified += $file.Length
        $filesToRemove += $file.FullName
        Write-Action "Backup file" $file.FullName
    }
    
    # Get .bak files
    $bakFiles = Get-ChildItem -Path . -Recurse -File | Where-Object { $_.Extension -eq ".bak" } 
    foreach ($file in $bakFiles) {
        $totalFilesIdentified++
        $totalSizeIdentified += $file.Length
        $filesToRemove += $file.FullName
        Write-Action "Backup file" $file.FullName
    }
}

# 2. Identify temporary/archive files
if ($performCleanTemp) {
    Write-Section "Identifying temporary and archive files..."
    
    # Archive files
    $archiveFiles = @(
        "arxiv_submission.tar",
        "arxiv_submission.tar.gz",
        "arxiv_submission_updated.tar.gz"
    )
    
    foreach ($file in $archiveFiles) {
        if (Test-Path $file) {
            $fileObj = Get-Item $file
            $totalFilesIdentified++
            $totalSizeIdentified += $fileObj.Length
            $filesToRemove += $fileObj.FullName
            Write-Action "Archive file" $fileObj.FullName
        }
    }
    
    # Previous backup directories (but not the one we just created)
    $backupDirs = Get-ChildItem -Path . -Directory | Where-Object { 
        $_.Name -like "backup_*" -and (-not $CreateBackup -or -not ($_.Name -like "safe_backup_$timestamp"))
    }
    
    foreach ($dir in $backupDirs) {
        $dirSize = (Get-ChildItem $dir.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum
        $totalFilesIdentified++
        $totalSizeIdentified += $dirSize
        $filesToRemove += $dir.FullName  # We'll handle directories specially when removing
        Write-Action "Backup directory" $dir.FullName
    }
}

# 3. Identify cache files
if ($performCleanCache) {
    Write-Section "Identifying cache files (preserving directory structure)..."
    
    # Main cache directories
    $cacheDirs = @(
        "cache",
        "visualization/cache",
        "concept_fragmentation/llm/cache"
    )
    
    foreach ($dir in $cacheDirs) {
        if (Test-Path $dir) {
            $cacheFiles = Get-ChildItem -Path $dir -Recurse -File | Where-Object { $_.Name -ne ".gitkeep" }
            foreach ($file in $cacheFiles) {
                $totalFilesIdentified++
                $totalSizeIdentified += $file.Length
                $filesToRemove += $file.FullName
                Write-Action "Cache file" $file.FullName
            }
        }
    }
}

# 4. Identify duplicate/outdated files
if ($performCleanDuplicates) {
    Write-Section "Identifying duplicate and outdated files..."
    
    $outdatedFiles = @(
        "concept_fragmentation/foundations_consolidated.md",
        "concept_fragmentation/foundations_update.md",
        "concept_fragmentation/foundations_update_llm.md",
        "concept_fragmentation/foundations_with_figures.md",
        "concept_fragmentation/analysis/cluster_paths_clean.py"
    )
    
    foreach ($file in $outdatedFiles) {
        if (Test-Path $file) {
            $fileObj = Get-Item $file
            $totalFilesIdentified++
            $totalSizeIdentified += $fileObj.Length
            $filesToRemove += $fileObj.FullName
            Write-Action "Outdated file" $fileObj.FullName
        }
    }
    
    # Check for potential duplicate scripts
    $dashboardUpdateFiles = @(
        "update_dashboard.py",
        "manual_dashboard_update.py",
        "refresh_dashboard.py"
    )
    
    # Keep the newest file
    $newestFile = $null
    $newestDate = [DateTime]::MinValue
    
    foreach ($file in $dashboardUpdateFiles) {
        if (Test-Path $file) {
            $fileObj = Get-Item $file
            if ($fileObj.LastWriteTime -gt $newestDate) {
                $newestFile = $file
                $newestDate = $fileObj.LastWriteTime
            }
        }
    }
    
    # Mark older duplicates for removal
    foreach ($file in $dashboardUpdateFiles) {
        if ((Test-Path $file) -and ($file -ne $newestFile)) {
            $fileObj = Get-Item $file
            $totalFilesIdentified++
            $totalSizeIdentified += $fileObj.Length
            $filesToRemove += $fileObj.FullName
            Write-Action "Duplicate dashboard update script" $fileObj.FullName
        }
    }
}

# 5. Identify Python cache files
if ($performCleanPycache) {
    Write-Section "Identifying Python cache files..."
    
    # __pycache__ directories
    $pycacheDirs = Get-ChildItem -Path . -Recurse -Directory -Force | Where-Object { $_.Name -eq "__pycache__" }
    foreach ($dir in $pycacheDirs) {
        $dirSize = (Get-ChildItem $dir.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum
        $totalFilesIdentified++
        $totalSizeIdentified += $dirSize
        $filesToRemove += $dir.FullName  # We'll handle directories specially when removing
        Write-Action "Python cache directory" $dir.FullName
    }
    
    # .pyc files
    $pycFiles = Get-ChildItem -Path . -Recurse -File | Where-Object { $_.Extension -eq ".pyc" }
    foreach ($file in $pycFiles) {
        $totalFilesIdentified++
        $totalSizeIdentified += $file.Length
        $filesToRemove += $file.FullName
        Write-Action "Python compiled file" $file.FullName
    }
}

# Summary of what was identified
Write-Title "Cleanup Summary"
Write-Host "Identified $totalFilesIdentified files/directories to remove" -ForegroundColor White
Write-Host "Total size: $($totalSizeIdentified / 1MB) MB" -ForegroundColor White

# If list-only mode, stop here
if ($ListOnly) {
    Write-Host "`nList-only mode completed. No files were modified." -ForegroundColor Yellow
    exit
}

# Ask for confirmation before proceeding with actual deletion
$confirmation = Read-Host "`nDo you want to proceed with removing these files? (y/n)"
if ($confirmation -ne 'y') {
    Write-Host "Cleanup canceled. No files were modified." -ForegroundColor Yellow
    exit
}

# Actually perform the deletions
Write-Section "Performing cleanup..."
$filesRemoved = 0
$sizeRemoved = 0

foreach ($path in $filesToRemove) {
    try {
        $item = Get-Item $path -ErrorAction SilentlyContinue
        if ($item) {
            if ($item.PSIsContainer) {
                # It's a directory
                Remove-Item $path -Recurse -Force -ErrorAction Stop
                Write-Host "  Removed directory: $path" -ForegroundColor Gray
            } else {
                # It's a file
                $sizeRemoved += $item.Length
                Remove-Item $path -Force -ErrorAction Stop
                Write-Host "  Removed file: $path" -ForegroundColor Gray
            }
            $filesRemoved++
        }
    } catch {
        Write-Host "  Error removing $path : $_" -ForegroundColor Red
    }
}

# Add .gitkeep files to cache directories if they've been cleaned
if ($performCleanCache) {
    $cacheDirs = @(
        "cache",
        "visualization/cache",
        "concept_fragmentation/llm/cache"
    )
    
    foreach ($dir in $cacheDirs) {
        if (Test-Path $dir) {
            # Add .gitkeep to main cache dir
            if (-not (Test-Path "$dir/.gitkeep")) {
                New-Item -ItemType File -Path "$dir/.gitkeep" -Force | Out-Null
                Write-Host "  Added .gitkeep to $dir" -ForegroundColor Gray
            }
            
            # Add .gitkeep to subdirectories
            Get-ChildItem -Path $dir -Directory | ForEach-Object {
                if (-not (Test-Path "$($_.FullName)/.gitkeep")) {
                    New-Item -ItemType File -Path "$($_.FullName)/.gitkeep" -Force | Out-Null
                    Write-Host "  Added .gitkeep to $($_.FullName)" -ForegroundColor Gray
                }
            }
        }
    }
}

# Verify dashboard still works if requested
if ($VerifyDashboard) {
    $dashboardWorks = Test-DashboardFunctionality
    if (-not $dashboardWorks) {
        Write-Host "`nWARNING: Dashboard verification failed! You may want to restore from backup." -ForegroundColor Red
        if ($CreateBackup) {
            Write-Host "  Backup is available in: $backupDir" -ForegroundColor Yellow
        }
    } else {
        Write-Host "`nDashboard verification successful!" -ForegroundColor Green
    }
}

# Final summary
Write-Title "Cleanup Completed"
Write-Host "Successfully removed $filesRemoved files/directories" -ForegroundColor Green
Write-Host "Total space cleared: $($sizeRemoved / 1MB) MB" -ForegroundColor Green

if ($CreateBackup) {
    Write-Host "`nA backup of critical files was created in: $backupDir" -ForegroundColor Yellow
    Write-Host "You can restore files from this backup if needed." -ForegroundColor Yellow
}

Write-Host "`nCleanup completed successfully!" -ForegroundColor Cyan