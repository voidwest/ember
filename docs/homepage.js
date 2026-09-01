(() => {
    const setup = () => {
        const body = document.body;
        const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
        if (body.classList.contains("home") && !reducedMotion.matches) {
            let pointer;
            let frame;
            const renderField = () => {
                const x = pointer.clientX / window.innerWidth - 0.5;
                const y = pointer.clientY / window.innerHeight - 0.5;
                body.style.setProperty("--field-shift-x", `${(x * 14).toFixed(2)}px`);
                body.style.setProperty("--field-shift-y", `${(y * 10).toFixed(2)}px`);
                body.style.setProperty("--network-shift-x", `${(-x * 18).toFixed(2)}px`);
                body.style.setProperty("--network-shift-y", `${(-y * 12).toFixed(2)}px`);
                frame = undefined;
            };
            window.addEventListener("pointermove", (event) => {
                if (event.pointerType === "touch") return;
                pointer = event;
                if (!frame) frame = requestAnimationFrame(renderField);
            }, { passive: true });
            document.documentElement.addEventListener("pointerleave", () => {
                if (frame) cancelAnimationFrame(frame);
                frame = undefined;
                body.style.setProperty("--field-shift-x", "0px");
                body.style.setProperty("--field-shift-y", "0px");
                body.style.setProperty("--network-shift-x", "0px");
                body.style.setProperty("--network-shift-y", "0px");
            });
        }

        const chart = document.querySelector(".contribution-chart");
        const image = chart?.querySelector("img");
        const tooltip = chart?.querySelector(".contribution-tooltip");
        if (!chart || !image || !tooltip) return;

        const arabic = document.documentElement.lang.startsWith("ar");
        const locale = arabic ? "ar-SA" : "en-US";
        const number = new Intl.NumberFormat(locale);
        const date = new Intl.DateTimeFormat(locale, {
            year: "numeric",
            month: "long",
            day: "numeric",
            timeZone: "UTC",
        });
        let graph;

        const loadGraph = async () => {
            if (graph) return graph;
            const response = await fetch(image.currentSrc || image.src);
            const document = new DOMParser().parseFromString(await response.text(), "image/svg+xml");
            const svg = document.documentElement;
            const viewBox = svg.viewBox.baseVal;
            const cells = [...document.querySelectorAll("rect[data-date]")].map((cell) => ({
                x: Number(cell.getAttribute("x")),
                y: Number(cell.getAttribute("y")),
                width: Number(cell.getAttribute("width")),
                height: Number(cell.getAttribute("height")),
                date: cell.dataset.date,
                count: Number(cell.dataset.count),
            }));
            graph = { width: viewBox.width, height: viewBox.height, cells };
            return graph;
        };

        const hide = () => tooltip.classList.remove("is-visible");
        const showCell = async (event) => {
            try {
                const data = await loadGraph();
                const imageBox = image.getBoundingClientRect();
                if (
                    event.clientX < imageBox.left || event.clientX > imageBox.right ||
                    event.clientY < imageBox.top || event.clientY > imageBox.bottom
                ) {
                    hide();
                    return;
                }
                const x = ((event.clientX - imageBox.left) / imageBox.width) * data.width;
                const y = ((event.clientY - imageBox.top) / imageBox.height) * data.height;
                const cell = data.cells.find((item) =>
                    x >= item.x && x <= item.x + item.width &&
                    y >= item.y && y <= item.y + item.height
                );
                if (!cell) {
                    hide();
                    return;
                }
                const day = date.format(new Date(`${cell.date}T00:00:00Z`));
                tooltip.textContent = arabic
                    ? `${number.format(cell.count)} مساهمة: ${day}`
                    : cell.count === 0
                        ? `No contributions on ${day}`
                        : `${number.format(cell.count)} contribution${cell.count === 1 ? "" : "s"} on ${day}`;
                const chartBox = chart.getBoundingClientRect();
                const left = Math.max(76, Math.min(chartBox.width - 76, event.clientX - chartBox.left));
                tooltip.style.left = `${left}px`;
                tooltip.style.top = `${event.clientY - chartBox.top - 8}px`;
                tooltip.classList.add("is-visible");
            } catch {
                hide();
            }
        };

        chart.addEventListener("pointermove", showCell);
        chart.addEventListener("pointerleave", hide);
        chart.addEventListener("blur", hide);
        loadGraph().catch(() => {});
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", setup, { once: true });
    } else {
        setup();
    }
})();
