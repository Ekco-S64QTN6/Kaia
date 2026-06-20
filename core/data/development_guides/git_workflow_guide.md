# Git Workflow Walkthrough

This guide covers the essential Git commands for managing your project's version control.

## I. Standard Git Workflow

For everyday changes and new features:

1.  **Make Your Changes:** Edit, add, or delete files in your working directory.

2.  **Stage Your Changes:** Tell Git which modified or new files you want to include in your next commit.
    ```bash
    git add <file(s)>
    # Example: git add README.md src/main.py
    # To stage all changes: git add .
    ```

3.  **Commit Your Changes:** Record the staged changes as a new commit in your local repository.
    ```bash
    git commit -m "Your concise commit message here"
    # For a multi-line message, just run `git commit` (opens editor)
    ```

4.  **Push Your Commits:** Upload your new local commits to the remote repository (e.g., GitHub).
    ```bash
    git push
    ```

## II. Amending the Last Commit

Use this when you need to modify the message of the very last commit or include additional changes that should have been part of that commit. This rewrites your local history for that specific commit.

1.  **Make Any New Changes (Optional):** If you have additional file changes you want to add to the last commit, make them now and save.

2.  **Stage New Changes (Optional):** If you made new changes in step 1, stage them. If you're only editing the commit message, you can skip this step or still run `git add .` (it's harmless if nothing new is changed).
    ```bash
    git add <file(s)>
    # Example: git add README.md
    ```

3.  **Amend the Last Commit:** This command will:
    * Open your default Git editor with the message of your last commit loaded.
    * If you staged new changes, they will be incorporated into this commit.
    * The old commit will be replaced by a new one with a different SHA-1 hash.
    ```bash
    git commit --amend
    ```
    * In the editor: Edit the commit message as desired. Save and close the editor (`:wq` in Vim, `Ctrl+X` then `Y` then `Enter` in Nano).

4.  **Force Push to Remote (Use with Caution!):** Since you've rewritten history locally, a regular `git push` won't work. You need to force the update.
    ```bash
    git push --force-with-lease
    ```
    * **Important Note:** `git push --force-with-lease` is safer than `--force` because it will fail if the remote branch has new commits you don't have locally, preventing accidental overwrites of others' work. Use force pushing only when you are certain it's safe (e.g., on a branch only you are working on).

## III. Undoing Changes

### 3.1. Unstaging Changes (`git reset`)

To unstage changes that have been added to the staging area but not yet committed:

```bash
git reset HEAD <file(s)>
# Example: git reset HEAD src/main.py
# To unstage all changes: git reset HEAD .

Effect: Moves changes from the staging area back to the working directory. The changes are kept, just no longer staged for the next commit.
