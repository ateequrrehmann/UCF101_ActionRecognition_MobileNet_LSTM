using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Text.Json;
using Microsoft.Extensions.Configuration;

namespace Frontend.Services
{
    public class ActionService
    {
        private readonly HttpClient _client;
        private readonly string _apiUrl;

        public ActionService(IConfiguration configuration)
        {
            _client = new HttpClient();
            _apiUrl = configuration["ActionApi:Url"] ?? "https://rehmanateequr501--action-recognition-api-predict-action-api.modal.run";
        }

        public async Task<(bool Success, string Message, string Action, double Confidence)> GetActionPredictionAsync(string filePath)
        {
            if (!File.Exists(filePath))
                return (false, "File not found.", null, 0);

            try
            {
                byte[] fileBytes = await File.ReadAllBytesAsync(filePath);

                using (var content = new ByteArrayContent(fileBytes))
                {
                    content.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");

                    HttpResponseMessage response = await _client.PostAsync(_apiUrl, content);

                    string jsonResponse = await response.Content.ReadAsStringAsync();

                    if (response.IsSuccessStatusCode)
                    {
                        using (JsonDocument doc = JsonDocument.Parse(jsonResponse))
                        {
                            string action = doc.RootElement.GetProperty("action").GetString();
                            double confidence = doc.RootElement.GetProperty("confidence").GetDouble();
                            return (true, "Success", action, confidence);
                        }
                    }
                    else
                    {
                        return (false, $"Server Error: {response.StatusCode}: {jsonResponse}", null, 0);
                    }
                }
            }
            catch (Exception ex)
            {
                return (false, $"Exception: {ex.Message}", null, 0);
            }
        }
    }
}
