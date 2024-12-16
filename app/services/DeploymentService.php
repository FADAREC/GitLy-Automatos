<?php

namespace App\Services;

use GuzzleHttp\Client;

class DeploymentService
{
    private $client;
    private $netlifyApiKey;

    public function __construct()
    {
        $this->client = new Client();
        $this->netlifyApiKey = getenv('NETLIFY_API_KEY');
    }

    public function deployToNetlify(string $siteId, string $directory)
    {
        $zipPath = "{$directory}.zip";

        exec("zip -r {$zipPath} {$directory}", $output, $status);

        if ($status !== 0) {
            return "Error compressing directory: " . implode("\n", $output);
        }

        $response = $this->client->post("https://api.netlify.com/api/v1/sites/{$siteId}/deploys", [
            'headers' => [
                'Authorization' => "Bearer {$this->netlifyApiKey}",
            ],
            'multipart' => [
                [
                    'name' => 'file',
                    'contents' => fopen($zipPath, 'r'),
                ],
            ],
        ]);

        unlink($zipPath);

        return $response->getStatusCode() === 200 ? "Deployment successful!" : "Deployment failed!";
    }
}