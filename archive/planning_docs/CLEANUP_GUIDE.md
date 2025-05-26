# Repository Cleanup Guide

This guide explains how to safely clean up the repository to remove temporary, backup, and unused files while preserving all functionality needed for the dashboard and paper generation.

## Quick Start

The safest way to clean the repository is to use the `safe_cleanup.ps1` script, which provides several options for selective cleanup and verification:

```powershell
# First, list what would be removed without actually deleting anything
./safe_cleanup.ps1 -ListOnly -All

# Clean only backup files and Python cache
./safe_cleanup.ps1 -CleanBackups -CleanPycache

# Clean everything and verify dashboard still works afterward
./safe_cleanup.ps1 -All -VerifyDashboard
```

## Cleanup Script Options

The script supports the following parameters:

| Parameter | Description |
|-----------|-------------|
| `-ListOnly` | Only list files that would be removed, don't delete anything |
| `-CleanBackups` | Remove backup files (*.bak, *.bak.*) |
| `-CleanCache` | Clean cache files (preserves directory structure) |
| `-CleanTemp` | Remove temporary files and old backup directories |
| `-CleanPycache` | Remove Python cache files (__pycache__, *.pyc) |
| `-CleanDuplicates` | Remove duplicate/outdated files |
| `-All` | Clean everything (equivalent to all the Clean* options) |
| `-VerifyDashboard` | Verify dashboard works after cleanup |
| `-CreateBackup` | Create backup before cleanup (default: true) |

## What Gets Cleaned

The script will clean the following categories of files:

### Backup Files
- All `.bak.*` files across the repository:
  - `/concept_fragmentation/foundations.md.bak.*` (all date-stamped versions)
  - `/visualization/dash_app.py.bak.*` (all dated backups)

### Temporary Files
- Old backup directories (backup_*)
- Archive files (*.tar, *.tar.gz)

### Cache Files
- Contents of cache directories (preserving the directory structure)
  - `/cache/`
  - `/visualization/cache/`
  - `/concept_fragmentation/llm/cache/`

### Duplicate/Outdated Files
- Various outdated foundation documents
- Duplicate dashboard update scripts

### Python Cache
- `__pycache__` directories
- `.pyc` files

## What's Preserved

The script carefully preserves:

- All core dashboard components
- Core analysis code
- LLM integration components
- Paper generation files
- Core runners and scripts
- Directory structure (especially for cache directories)

## Safety Features

The script includes several safety features:

1. **List-only mode**: Preview what would be deleted without actually removing anything
2. **Comprehensive backup**: Creates a timestamped backup of critical files before cleanup
3. **Category-based cleanup**: Choose which types of files to clean
4. **Dashboard verification**: Verify the dashboard still works after cleanup
5. **Restore capability**: Backups allow restoration if needed

## Manual Cleanup

If you prefer to clean up manually, follow these general guidelines:

1. **Backup first**: Create backups of critical files
2. Start with obvious temporary files:
   - `.bak.*` files
   - Old backup directories
3. Clean cache directories but preserve their structure
4. Remove Python cache files
5. Test the dashboard after cleanup

## Troubleshooting

If the dashboard stops working after cleanup:

1. Check the error message to identify which import might be failing
2. Restore critical files from the backup created during cleanup
3. If all else fails, restore the entire repository from version control

Remember that a clean repository improves navigation, reduces confusion, and helps new contributors understand the project structure more easily.