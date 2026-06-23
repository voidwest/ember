(function() {
    var key = "voidwest-theme";
    var root = document.documentElement;

    function storedTheme() {
        try {
            return localStorage.getItem(key);
        } catch (_) {
            return null;
        }
    }

    function saveTheme(theme) {
        try {
            localStorage.setItem(key, theme);
        } catch (_) {}
    }

    function applyTheme(theme) {
        root.dataset.theme = theme === "light" ? "light" : "dark";
        document.querySelectorAll(".theme-toggle").forEach(function(button) {
            var active = root.dataset.theme;
            var next = active === "light" ? "dark" : "light";
            button.textContent = next;
            button.setAttribute("aria-pressed", active === "light" ? "true" : "false");
            button.setAttribute("aria-label", "Switch to " + next + " theme");
            button.title = "Switch to " + next + " theme";
        });
    }

    applyTheme(storedTheme() || "dark");

    document.addEventListener("DOMContentLoaded", function() {
        applyTheme(root.dataset.theme);
        document.querySelectorAll(".theme-toggle").forEach(function(button) {
            button.addEventListener("click", function() {
                var next = root.dataset.theme === "light" ? "dark" : "light";
                applyTheme(next);
                saveTheme(next);
            });
        });
    });
})();
