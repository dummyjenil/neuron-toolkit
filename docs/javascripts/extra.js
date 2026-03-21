document$.subscribe(() => {
    const canvas = document.getElementById("drawCanvas") || Object.assign(document.body.appendChild(document.createElement("canvas")), { id: "drawCanvas" });

    const resize = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
    addEventListener("resize", resize);
    resize();

    function initMascot() {
        let mascot = document.getElementById("mascot") || Object.assign(document.body.appendChild(document.createElement("div")), {
            id: "mascot",
            innerText: "🧸",
            style: "left:50px;bottom:50px"
        });

        let catches = 0;
        mascot.addEventListener("mouseenter", () => {
            if (catches < 8) {
                gsap.to(mascot, {
                    left: Math.random() * (innerWidth - 100),
                    top: Math.random() * (innerHeight - 100),
                    duration: 0.3,
                    ease: "back.out(2)"
                });
                catches++;
            } else {
                mascot.innerText = "👑";
                gsap.to(mascot, { scale: 1.5, rotation: 360, duration: 1 });
            }
        });
    }

    const colors = ['#ff6b6b', '#ffd43b', '#4dabf7', '#63e6be', '#91a7ff'];
    addEventListener("mousedown", e => {
        for (let i = 0; i < 6; i++) {
            const drop = Object.assign(document.body.appendChild(document.createElement("div")), { className: "ink-drop" });
            const size = Math.random() * 25 + 10;

            Object.assign(drop.style, {
                width: size + "px",
                height: size + "px",
                backgroundColor: colors[Math.random() * colors.length | 0],
                borderRadius: "50%",
                left: e.clientX + "px",
                top: e.clientY + "px"
            });

            gsap.to(drop, {
                x: (Math.random() - 0.5) * 200,
                y: (Math.random() - 0.5) * 200,
                scale: Math.random() * 2,
                opacity: 0,
                duration: 1.5,
                ease: "power2.out",
                onComplete: () => drop.remove()
            });
        }
    });

    document.querySelectorAll(".md-typeset a, button").forEach(btn => {
        btn.addEventListener("mousemove", e => {
            const r = btn.getBoundingClientRect();
            gsap.to(btn, {
                x: (e.clientX - r.left - r.width / 2) * 0.4,
                y: (e.clientY - r.top - r.height / 2) * 0.4,
                duration: 0.3
            });
        });
        btn.addEventListener("mouseleave", () => gsap.to(btn, { x: 0, y: 0, duration: 0.5, ease: "elastic.out(1,0.3)" }));
    });

    gsap.from(".md-main__inner", { opacity: 0, y: 50, rotation: -2, duration: 1, ease: "power4.out" });

    initMascot();
});

document.querySelectorAll(".md-typeset pre code").forEach(codeBlock => {
    const container = codeBlock.parentElement;
    Object.assign(container.style, {
        filter: "blur(12px)",
        transition: "filter 0.6s cubic-bezier(0.4,0,0.2,1)",
        cursor: "zoom-in"
    });

    const tokens = codeBlock.querySelectorAll("span");
    tokens.length ? tokens.forEach(t => t.style.opacity = "0") : codeBlock.style.opacity = "0";

    container.addEventListener("click", function () {
        if (this.dataset.revealed) return;
        this.dataset.revealed = true;

        Object.assign(this.style, { filter: "blur(0)", cursor: "default" });

        if (canPlayAudio) scribbleSound.play().catch(()=>{});

        tokens.length
            ? gsap.to(tokens, {
                opacity: 1,
                duration: 0.1,
                stagger: 0.05,
                ease: "none",
                onComplete: () => {
                    scribbleSound.pause();
                    scribbleSound.currentTime = 0;
                    scribbleSound.volume = 0.3;
                }
            })
            : gsap.to(codeBlock, { opacity: 1, duration: 1 });

    }, { once: true });
});

gsap.registerPlugin(ScrollTrigger, ScrollToPlugin);

document.querySelectorAll(".md-typeset p, .md-typeset li").forEach(el => {
    gsap.from(el, {
        scrollTrigger: { trigger: el, start: "top 90%", toggleActions: "play none none none" },
        opacity: 0,
        y: 20,
        duration: 0.8,
        ease: "power2.out"
    });
});

document.addEventListener("DOMContentLoaded", () => {
    const main = document.querySelector(".md-main__inner");

    gsap.fromTo(main,
        { rotationY: -90, opacity: 0, transformOrigin: "left center" },
        { rotationY: 0, opacity: 1, duration: 1.2, ease: "back.out(1.5)" }
    );

    document.querySelectorAll(".md-nav__link, .md-typeset a").forEach(link => {
        link.addEventListener("click", e => {
            const href = link.getAttribute("href");
            if (!href || href.startsWith("#") || href.startsWith("http")) return;

            e.preventDefault();
            gsap.to(main, {
                rotationY: 90,
                opacity: 0,
                duration: 0.6,
                onStart: () => playSFX('page-flip', 0.5),
                onComplete: () => location.href = href
            });
        });
    });
});

const scribbleSound = Object.assign(new Audio('/sound/pancil_writing.mp3'), { loop: true, volume: 0.3 });

document$.subscribe(() => {
    document.querySelectorAll('a[href*="#"]').forEach(link => {
        link.addEventListener("click", e => {
            const id = link.getAttribute("href").split("#")[1];
            const target = document.querySelector("#" + id);
            if (!target) return;

            e.preventDefault();
            e.stopPropagation();

            const mascot = document.getElementById("mascot");
            if (mascot) gsap.to(mascot, { scale: 1.8, rotation: 15, duration: 0.3 });

            gsap.to(window, {
                duration: 2,
                scrollTo: { y: target, offsetY: 80, autoKill: false },
                ease: "elastic.out(1,0.6)",
                overwrite: true,
                onStart: () => playSFX('slide-whistle', 0.3),
                onComplete: () => {
                    if (mascot) gsap.to(mascot, { scale: 1, rotation: 0, duration: 0.4 });
                    gsap.fromTo(target, { scale: 0.9, rotation: -2 }, { scale: 1, rotation: 0, duration: 0.8, ease: "back.out(4)" });
                }
            });
        });
    });
});

document.querySelectorAll(".md-nav__link, .md-footer__link").forEach(el => {
    el.addEventListener("mouseenter", () => gsap.to(el, {
        x: Math.random() * 4 - 2,
        rotation: Math.random() * 2 - 1,
        duration: 0.1,
        repeat: 5,
        yoyo: true
    }));
    el.addEventListener("mouseleave", () => gsap.to(el, { x: 0, rotation: 0, duration: 0.3 }));
});

let canPlayAudio = false;
document.addEventListener('click', () => canPlayAudio = true, { once: true });

const playSFX = (file, volume = 0.4) => {
    if (!canPlayAudio) return;
    const audio = new Audio(`/sound/${file}.mp3`);
    audio.volume = volume;
    audio.play().catch(()=>{});
};