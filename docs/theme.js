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
            var label = root.lang === "ar"
                ? "التبديل إلى الوضع " + (next === "light" ? "الفاتح" : "الداكن")
                : "Switch to " + next + " theme";
            button.setAttribute("aria-label", label);
            button.title = label;
        });
    }

    function enhanceSiteNavigation() {
        document.querySelectorAll(".site-nav").forEach(function(nav, index) {
            var menu = nav.querySelector(".nav-links");
            var actions = nav.querySelector(".nav-actions");
            var themeButton = actions && actions.querySelector(".theme-toggle");
            if (!menu || !actions || nav.querySelector(".menu-toggle")) return;

            if (!menu.id) menu.id = "site-menu-" + index;

            if (themeButton) {
                var mobileTheme = themeButton.cloneNode(true);
                mobileTheme.classList.add("mobile-theme-toggle");
                menu.appendChild(mobileTheme);
            }

            var button = document.createElement("button");
            button.className = "menu-toggle";
            button.type = "button";
            button.setAttribute("aria-label", root.lang === "ar" ? "فتح قائمة التنقل" : "Open navigation");
            button.setAttribute("aria-controls", menu.id);
            button.setAttribute("aria-expanded", "false");
            button.innerHTML = "<span></span><span></span><span></span>";
            actions.appendChild(button);
        });
    }

    applyTheme(storedTheme() || "dark");

    document.addEventListener("DOMContentLoaded", function() {
        enhanceSiteNavigation();
        applyTheme(root.dataset.theme);
        document.querySelectorAll(".theme-toggle").forEach(function(button) {
            button.addEventListener("click", function() {
                var next = root.dataset.theme === "light" ? "dark" : "light";
                applyTheme(next);
                saveTheme(next);
            });
        });

        document.querySelectorAll(".menu-toggle").forEach(function(button) {
            var menu = document.getElementById(button.getAttribute("aria-controls"));
            if (!menu) return;

            function closeMenu() {
                button.setAttribute("aria-expanded", "false");
                button.setAttribute("aria-label", root.lang === "ar" ? "فتح قائمة التنقل" : "Open navigation");
                menu.removeAttribute("data-open");
            }

            button.addEventListener("click", function() {
                var opening = button.getAttribute("aria-expanded") !== "true";
                button.setAttribute("aria-expanded", opening ? "true" : "false");
                button.setAttribute("aria-label", root.lang === "ar"
                    ? (opening ? "إغلاق قائمة التنقل" : "فتح قائمة التنقل")
                    : (opening ? "Close navigation" : "Open navigation"));
                if (opening) menu.setAttribute("data-open", "true");
                else menu.removeAttribute("data-open");
            });

            menu.addEventListener("click", function(event) {
                if (event.target.closest("a")) closeMenu();
            });

            document.addEventListener("keydown", function(event) {
                if (event.key === "Escape") closeMenu();
            });
        });
    });
})();
