<script>
  const form = document.getElementById("uploadForm");

  if (form) {
    form.addEventListener("submit", async function (e) {
      e.preventDefault();

      const formData = new FormData(form);

      const uploadRes = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      if (uploadRes.ok) {
        const transcriptionRes = await fetch("/transcribe", { method: "POST" });
        const transcriptionData = await transcriptionRes.json();
        document.getElementById("transcription").innerText = transcriptionData.transcription;

        const emotionRes = await fetch("/predict", { method: "POST" });
        const emotionData = await emotionRes.json();
        document.getElementById("emotion").innerText = emotionData.emotion;
      } else {
        alert("Error uploading the audio file.");
      }
    });
  }
</script>
