const aiData = document.getElementById("ai-data");

async function updateData() {
    const status = document.getElementById("data-status");

    try {
        const response = await fetch("/data", {
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error("HTTP error");
        }

        const data = await response.json();

        if (Object.keys(data).length === 0) {
            status.textContent = "WAITING";
            status.className = "status-waiting";

            aiData.textContent =
                "Đang chờ dữ liệu từ AI...";

            document.getElementById("face-name").textContent = "--";
            document.getElementById("face-message").textContent =
                "Chưa có dữ liệu nhận diện";

            document.getElementById("confidence").textContent = "--";
            document.getElementById("face-count").textContent = "0";

            return;
        }

        aiData.textContent = JSON.stringify(data, null, 2);

        document.getElementById("update-time").textContent =
            data.time || "--:--:--";

        const name = data.name || data.person || "--";

        document.getElementById("face-name").textContent = name;

        document.getElementById("face-message").textContent =
            name !== "--"
                ? "Đã nhận được kết quả nhận diện"
                : "Đã nhận dữ liệu JSON";

        const confidence =
            data.confidence ?? data.confidence_score;

        let confidenceText = "--";

        if (confidence !== undefined && confidence !== null) {
            if (typeof confidence === "number") {
                confidenceText =
                    (confidence <= 1
                        ? confidence * 100
                        : confidence
                    ).toFixed(1) + "%";
            } else {
                confidenceText = confidence;
            }
        }

        document.getElementById("confidence").textContent =
            confidenceText;

        document.getElementById("face-count").textContent =
            data.count ?? data.face_count ?? 0;

        status.textContent = "CONNECTED";
        status.className = "";

    } catch (error) {
        status.textContent = "DISCONNECTED";
        status.className = "status-disconnected";
        console.error("Loi nhan du lieu:", error);
    }
}

updateData();
setInterval(updateData, 1000);


// Trang thai camera
const camera = document.getElementById("camera");
const message = document.getElementById("camera-message");

camera.onload = function() {
    message.style.display = "none";
};

camera.onerror = function() {
    message.style.display = "block";
    message.textContent = "ĐANG CHỜ CAMERA...";
};