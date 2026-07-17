(function () {
    "use strict";

    var storageKey = "voidwest-theme";
    var root = document.documentElement;
    var systemPreference = window.matchMedia
        ? window.matchMedia("(prefers-color-scheme: dark)")
        : null;

    root.classList.add("js");

    function storedTheme() {
        try {
            var value = localStorage.getItem(storageKey);
            return value === "light" || value === "dark" ? value : null;
        } catch (_) {
            return null;
        }
    }

    function saveTheme(theme) {
        try {
            localStorage.setItem(storageKey, theme);
        } catch (_) {
            /* The toggle still works when storage is unavailable. */
        }
    }

    function systemTheme() {
        return systemPreference && systemPreference.matches ? "dark" : "light";
    }

    function isArabic() {
        return root.lang.toLowerCase().indexOf("ar") === 0;
    }

    function updateThemeControls() {
        var active = root.dataset.theme || systemTheme();
        var next = active === "dark" ? "light" : "dark";
        var arabic = isArabic();
        var text = arabic
            ? next === "dark"
                ? "داكن"
                : "فاتح"
            : next;
        var label = arabic
            ? next === "dark"
                ? "استخدام الوضع الداكن"
                : "استخدام الوضع الفاتح"
            : "Use " + next + " theme";

        document.querySelectorAll(".theme-toggle").forEach(function (button) {
            button.textContent = text;
            button.setAttribute("aria-label", label);
            button.setAttribute("aria-pressed", active === "dark" ? "true" : "false");
            button.setAttribute("title", label);
            button.dataset.themeTarget = next;
        });
    }

    function applyTheme(theme) {
        root.dataset.theme = theme === "dark" ? "dark" : "light";
        root.style.colorScheme = root.dataset.theme;
        updateThemeControls();
    }

    /* This runs synchronously in <head>, before page content is painted. */
    applyTheme(storedTheme() || systemTheme());

    function sectionLabel(heading) {
        return heading.textContent.replace(/\s*#\s*$/, "").trim();
    }

    function addHeadingAnchors(main) {
        main.querySelectorAll("h2[id], h3[id]").forEach(function (heading) {
            if (heading.querySelector(".heading-anchor")) {
                return;
            }

            var label = sectionLabel(heading);
            var anchor = document.createElement("a");
            anchor.className = "heading-anchor";
            anchor.href = "#" + heading.id;
            anchor.textContent = "#";
            anchor.setAttribute(
                "aria-label",
                isArabic() ? "رابط إلى قسم " + label : "Link to section: " + label,
            );
            heading.appendChild(anchor);
        });
    }

    function buildTableOfContents(main) {
        if (!main.matches(".document, .dossier")) {
            return;
        }

        var headings = Array.prototype.slice.call(
            main.querySelectorAll(":scope > h2[id]"),
        );
        if (headings.length < 4) {
            return;
        }

        var toc = document.createElement("aside");
        toc.className = "page-toc";
        toc.setAttribute("aria-label", isArabic() ? "في هذه الصفحة" : "On this page");

        var label = document.createElement("span");
        label.className = "toc-label";
        label.textContent = isArabic() ? "في هذه الصفحة" : "On this page";
        toc.appendChild(label);

        var list = document.createElement("ol");
        headings.forEach(function (heading) {
            var item = document.createElement("li");
            var link = document.createElement("a");
            link.href = "#" + heading.id;
            link.textContent = sectionLabel(heading);
            item.appendChild(link);
            list.appendChild(item);
        });
        toc.appendChild(list);
        main.insertBefore(toc, main.firstChild);

        if (!("IntersectionObserver" in window)) {
            return;
        }

        var links = toc.querySelectorAll("a");
        var linkById = {};
        links.forEach(function (link) {
            linkById[decodeURIComponent(link.hash.slice(1))] = link;
        });

        var observer = new IntersectionObserver(
            function (entries) {
                entries.forEach(function (entry) {
                    if (!entry.isIntersecting) {
                        return;
                    }
                    links.forEach(function (link) {
                        link.removeAttribute("aria-current");
                    });
                    if (linkById[entry.target.id]) {
                        linkById[entry.target.id].setAttribute(
                            "aria-current",
                            "location",
                        );
                    }
                });
            },
            {
                rootMargin: "-20% 0px -72% 0px",
                threshold: 0,
            },
        );

        headings.forEach(function (heading) {
            observer.observe(heading);
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        updateThemeControls();

        document.querySelectorAll(".theme-toggle").forEach(function (button) {
            button.addEventListener("click", function () {
                var next = root.dataset.theme === "dark" ? "light" : "dark";
                applyTheme(next);
                saveTheme(next);
            });
        });

        var main = document.querySelector("main");
        if (main) {
            addHeadingAnchors(main);
            buildTableOfContents(main);
        }

        root.classList.add("theme-ready");
    });

    if (systemPreference) {
        systemPreference.addEventListener("change", function () {
            if (!storedTheme()) {
                applyTheme(systemTheme());
            }
        });
    }
})();
