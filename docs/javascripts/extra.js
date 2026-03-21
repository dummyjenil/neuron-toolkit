document$.subscribe(() => {

    /* =========================
       🎮 CANVAS & XP SETUP
    ========================= */
    const canvas = document.getElementById("drawCanvas") || document.createElement("canvas");
    if (!canvas.id) {
        canvas.id = "drawCanvas";
        document.body.appendChild(canvas);
    }
    const ctx = canvas.getContext("2d");

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener("resize", resize);
    resize();

    let xp = parseInt(localStorage.getItem("xp") || 0);
    const updateXP = (amt = 0) => {
        if (amt > 0) playSFX('xp-collect', 0.4); // <--- ADD THIS
        xp += amt;
        localStorage.setItem("xp", xp);
        const xpBox = document.getElementById("xpBox") || document.createElement("div");
        if (!xpBox.id) { xpBox.id = "xpBox"; document.body.appendChild(xpBox); }
        xpBox.innerText = `Level ${Math.floor(xp / 100)} | ${xp} XP 🧠`;
    };
    updateXP();

    /* =========================
       🤖 MASCOT GAME (Fixed!)
    ========================= */
    function initMascot() {
        let mascot = document.getElementById("mascot");
        if (!mascot) {
            mascot = document.createElement("div");
            mascot.id = "mascot";
            mascot.innerText = "🧸";
            mascot.style.left = "50px";
            mascot.style.bottom = "50px";
            document.body.appendChild(mascot);
        }

        let catches = 0;
        mascot.addEventListener("mouseenter", () => {
            if (catches < 8) {
                const x = Math.random() * (window.innerWidth - 100);
                const y = Math.random() * (window.innerHeight - 100);
                gsap.to(mascot, { left: x, top: y, duration: 0.3, ease: "back.out(2)" });
                catches++;
                updateXP(5);
            } else {
                mascot.innerText = "👑";
                gsap.to(mascot, { scale: 1.5, rotation: 360, duration: 1 });
            }
        });
    }

    /* =========================
       🌈 HOLI MODE (Splatter)
    ========================= */
    const colors = ['#ff6b6b', '#ffd43b', '#4dabf7', '#63e6be', '#91a7ff'];

    window.addEventListener("mousedown", (e) => {
        for (let i = 0; i < 6; i++) {
            const drop = document.createElement("div");
            drop.className = "ink-drop";
            const size = Math.random() * 25 + 10;

            Object.assign(drop.style, {
                width: `${size}px`,
                height: `${size}px`,
                backgroundColor: colors[Math.floor(Math.random() * colors.length)],
                borderRadius: "50%",
                left: `${e.clientX}px`,
                top: `${e.clientY}px`
            });

            document.body.appendChild(drop);

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

    /* =========================
       ✨ MICRO-ANIMATIONS
    ========================= */
    // Magnetic Buttons
    document.querySelectorAll(".md-typeset a, button").forEach(btn => {
        btn.addEventListener("mousemove", (e) => {
            const rect = btn.getBoundingClientRect();
            gsap.to(btn, {
                x: (e.clientX - rect.left - rect.width / 2) * 0.4,
                y: (e.clientY - rect.top - rect.height / 2) * 0.4,
                duration: 0.3
            });
        });
        btn.addEventListener("mouseleave", () => {
            gsap.to(btn, { x: 0, y: 0, duration: 0.5, ease: "elastic.out(1, 0.3)" });
        });
    });

    // Page Intro
    gsap.from(".md-main__inner", { opacity: 0, y: 50, rotation: -2, duration: 1, ease: "power4.out" });

    initMascot();
});


/* =========================
   📝 PENCIL TYPING & BLUR EFFECT
   ========================= */
/* ==============================================
   🐍 PRO PYTHON TYPING WITH SYNTAX HIGHLIGHTING
   ============================================== */
document.querySelectorAll(".md-typeset pre code").forEach((codeBlock) => {
    const container = codeBlock.parentElement;

    // 1. Initial State
    container.style.filter = "blur(12px)";
    container.style.transition = "filter 0.6s cubic-bezier(0.4, 0, 0.2, 1)";
    container.style.cursor = "zoom-in";

    // Sabhi tokens (words) ko hide kar do
    const tokens = codeBlock.querySelectorAll("span");
    if (tokens.length > 0) {
        tokens.forEach(t => t.style.opacity = "0");
    } else {
        // Agar single text node hai toh character wrapping ki zaroorat padegi
        codeBlock.style.opacity = "0";
    }

    container.addEventListener("click", function () {
        if (this.dataset.revealed === "true") return;
        this.dataset.revealed = "true";

        // 2. Reveal Container
        this.style.filter = "blur(0px)";
        this.style.cursor = "default";

        if (typeof canPlayAudio !== 'undefined' && canPlayAudio) {
            scribbleSound.play().catch(e => console.log("Audio play blocked"));
        }
        // 3. Sequential Reveal (Typing Effect)
        if (tokens.length > 0) {
            gsap.to(tokens, {
                opacity: 1,
                duration: 0.1,
                stagger: 0.05, // Ek-ek word type hone ki speed
                ease: "none",
                onComplete: () => {
                    scribbleSound.pause();
                    scribbleSound.currentTime = 0; // Reset for next block
                    scribbleSound.volume = 0.3; // Reset volume for next use
                }
            });
        } else {
            gsap.to(codeBlock, { opacity: 1, duration: 1 });
        }

        updateXP(15);
    }, { once: true });
});


/* =========================
   📜 SCROLL REVEAL ANIMATION
   ========================= */
gsap.registerPlugin(ScrollTrigger);

document.querySelectorAll(".md-typeset p, .md-typeset li").forEach((el) => {
    gsap.from(el, {
        scrollTrigger: {
            trigger: el,
            start: "top 90%",
            toggleActions: "play none none none"
        },
        opacity: 0,
        y: 20,
        duration: 0.8,
        ease: "power2.out"
    });
});


/* =========================
   📖 PAGE FLIP ANIMATION
   ========================= */
document.addEventListener("DOMContentLoaded", () => {
    const mainContent = document.querySelector(".md-main__inner");

    // 1. Page Enter Animation (Jab naya page khule)
    gsap.fromTo(mainContent,
        { rotationY: -90, opacity: 0, transformOrigin: "left center" },
        { rotationY: 0, opacity: 1, duration: 1.2, ease: "back.out(1.5)" }
    );

    // 2. Page Exit Animation (Jab link click ho)
    document.querySelectorAll(".md-nav__link, .md-typeset a").forEach(link => {
        link.addEventListener("click", (e) => {
            const href = link.getAttribute("href");

            // External links ya anchor tags ko skip karo
            if (!href || href.startsWith("#") || href.startsWith("http")) return;

            e.preventDefault(); // Default load roko

            // Exit Animation Section mein:
            gsap.to(mainContent, {
                rotationY: 90,
                opacity: 0,
                duration: 0.6,
                onStart: () => playSFX('page-flip', 0.5), // <--- ADD THIS
                onComplete: () => {
                    window.location.href = href;
                }
            });
        });
    });
});


/* ==========================================
   🚀 BOUNCY SLIDE - FIXED FOR MKDOCS
   ========================================== */
gsap.registerPlugin(ScrollToPlugin);
// Scribble sound ka global object
const scribbleSound = new Audio('/sound/pancil_writing.mp3');
scribbleSound.loop = true; // Taaki typing ke beech mein sound ruke nahi
scribbleSound.volume = 0.3; // Background noise ki tarah halka rakho
// MkDocs Material ke Instant Loading ke liye ye wrapper zaroori hai
document$.subscribe(() => {

    // Saare internal links ko capture karo
    const jumpLinks = document.querySelectorAll('a[href*="#"]');

    jumpLinks.forEach(link => {
        link.addEventListener("click", function (e) {
            // URL se ID extract karo (e.g., #patterndetector)
            const href = this.getAttribute("href");
            const targetId = href.substring(href.indexOf("#"));
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                e.preventDefault();
                e.stopPropagation();

                // 1. Childish "Ready" Effect (Mascot Pop)
                const mascot = document.getElementById("mascot");
                if (mascot) gsap.to(mascot, { scale: 1.8, rotation: 15, duration: 0.3 });

                // 2. The Big Slide with Elastic Landing
                gsap.to(window, {
                    duration: 2, // Thoda slow taaki "slide" feel ho
                    scrollTo: {
                        y: targetElement,
                        offsetY: 80, // Sticky header ke liye gap
                        autoKill: false // User scroll se animation interrupt na ho
                    },
                    ease: "elastic.out(1, 0.6)", // Ye hai asli childish bounce!
                    overwrite: true,
                    onStart: () => playSFX('slide-whistle', 0.3),
                    onComplete: () => {
                        if (mascot) gsap.to(mascot, { scale: 1, rotation: 0, duration: 0.4 });
                        gsap.fromTo(targetElement,
                            { scale: 0.9, rotation: -2 },
                            { scale: 1, rotation: 0, duration: 0.8, ease: "back.out(4)" }
                        );
                    }
                });
            }
        });
    });
});



document.querySelectorAll(".md-nav__link, .md-footer__link").forEach(el => {
    el.addEventListener("mouseenter", () => {
        gsap.to(el, {
            x: Math.random() * 4 - 2,
            rotation: Math.random() * 2 - 1,
            duration: 0.1,
            repeat: 5,
            yoyo: true
        });
    });
    el.addEventListener("mouseleave", () => {
        gsap.to(el, { x: 0, rotation: 0, duration: 0.3 });
    });
});


let canPlayAudio = false;
document.addEventListener('click', () => { canPlayAudio = true; }, { once: true });
const playSFX = (file, volume = 0.4) => {
    if (!canPlayAudio) return;
    const audio = new Audio(`/sound/${file}.mp3`); // Example sound
    audio.volume = volume;
    audio.play().catch(() => { });
};