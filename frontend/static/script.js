const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const placeholderText = document.getElementById('placeholderText');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent = document.getElementById('resultContent');
const predClass = document.getElementById('predClass');
const predConf = document.getElementById('predConf');
const heatmapImage = document.getElementById('heatmapImage');

let selectedFile = null;

imageInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            placeholderText.classList.add('hidden');
        }
        reader.readAsDataURL(selectedFile);
    }
});

async function uploadImage() {
    if (!selectedFile) {
        alert("Please select an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    resultPlaceholder.textContent = "Processing image and generating Grad-CAM heatmap...";
    resultPlaceholder.classList.remove('hidden');
    resultContent.classList.add('hidden');

    try {
        const response = await fetch('http://127.0.0.1:8000/predict/', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log("Server Response:", data); // Check your F12 console to see exact keys

        if (response.ok) {
            resultPlaceholder.classList.add('hidden');
            resultContent.classList.remove('hidden');
            
            // Adjust keys here if your backend uses different JSON names (e.g., data.class_name or data.heatmap)
            predClass.textContent = data.prediction || data.class_name || "Unknown";
            predConf.textContent = data.confidence ? (data.confidence * 100).toFixed(2) + "%" : "N/A";
            
            if (data.heatmap_url) {
                heatmapImage.src = `http://127.0.0.1:8000${data.heatmap_url}`;
            }
        } else {
            alert("Error: " + (data.detail || "Unknown server error"));
            resultPlaceholder.textContent = "Failed to analyze image.";
        }
    } catch (err) {
        console.error("Fetch Error:", err);
        alert("Failed to process the response from the server.");
        resultPlaceholder.textContent = "Server response error.";
    }
}