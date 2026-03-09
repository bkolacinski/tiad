document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("convertForm");

    // Zapisywane pola
    const fields = ['title', 'orientation', 'fontSize', 'spacing', 'pageNumbers', 'formatPdf', 'formatDocx'];

    // Load from local storage
    fields.forEach(field => {
        const value = localStorage.getItem(field);
        if (value !== null) {
            const el = document.getElementById(field);
            if (el) {
                if (el.type === 'checkbox' || el.type === 'radio') {
                    el.checked = (value === 'true');
                } else {
                    el.value = value;
                }
            }
        }
    });

    // Save to local storage on change/submit
    form.addEventListener("change", (e) => {
        if (fields.includes(e.target.id) || e.target.name === 'format') {
            saveState();
        }
    });

    function saveState() {
        fields.forEach(field => {
            const el = document.getElementById(field);
            if (el) {
                if (el.type === 'checkbox' || el.type === 'radio') {
                    localStorage.setItem(field, el.checked);
                } else {
                    localStorage.setItem(field, el.value);
                }
            }
        });
    }

    // Dodatkowe zabezpieczenie form przed submit bez pliku, o ile HTML5 required by zawiodło
    form.addEventListener("submit", (e) => {
        const fileInput = document.getElementById("file");
        if (fileInput.files.length === 0) {
            e.preventDefault();
            alert("Proszę wybrać plik .xlsx przed konwersją.");
            return;
        }
        saveState();
    });
});
