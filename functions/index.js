const { onObjectFinalized } = require("firebase-functions/v2/storage");
const admin = require("firebase-admin");
const axios = require("axios");

admin.initializeApp();

exports.onNewSubtitleUpload = onObjectFinalized(async (event) => {
  const filePath = event.data.name;

  if (!filePath.endsWith(".srt") || !filePath.startsWith("subtitles/")) {
    console.log("Не е .srt файл или не е в subtitles/ папката.");
    return;
  }

  console.log(`🆕 Нов субтитър открит: ${filePath}`);

  const BACKEND_URL = "https://movie-app-backend-8emc.onrender.com/sync";

  return axios.post(BACKEND_URL, {
    filename: filePath,
  }, {
    headers: {
      "Content-Type": "application/json"
    }
  }).then(res => {
    console.log("📡 Успешно изпратена заявка:", res.data);
    return { success: true };
  }).catch(error => {
    console.error("❌ Грешка при изпращане към бекенда:", error.message);
    return { success: false };
  });
});