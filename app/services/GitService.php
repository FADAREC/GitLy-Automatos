<?php

namespace App\Services;

class GitService
{
    public function initRepository()
    {
        exec('git init', $output, $status);
        return $status === 0 ? "Git repository initialized!" : "Error initializing repository!";
    }

    public function addFiles(string $files = '.')
    {
        exec("git add {$files}", $output, $status);
        return $status === 0 ? "Files added successfully!" : "Error adding files!";
    }

    public function commitChanges(string $message)
    {
        exec("git commit -m \"{$message}\"", $output, $status);
        return $status === 0 ? "Changes committed!" : "Error committing changes!";
    }

    public function pushChanges(string $branch = 'main')
    {
        exec("git push origin {$branch}", $output, $status);
        return $status === 0 ? "Changes pushed to {$branch}!" : "Error pushing changes!";
    }
}