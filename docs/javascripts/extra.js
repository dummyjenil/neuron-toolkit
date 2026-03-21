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

        if (canPlayAudio) scribbleSound.play().catch(() => { });

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
    audio.play().catch(() => { });
};


// Scroll speed ke basis par paper ko bend karna
let proxy = { skew: 0 },
    skewSetter = gsap.quickSetter(".md-main__inner", "skewY", "deg"),
    clamp = gsap.utils.clamp(-5, 5); // Max 5 degree skew

ScrollTrigger.create({
    onUpdate: (self) => {
        let skew = clamp(self.getVelocity() / -300);
        if (Math.abs(skew) > Math.abs(proxy.skew)) {
            proxy.skew = skew;
            gsap.to(proxy, { skew: 0, duration: 0.8, ease: "power3", overwrite: true, onUpdate: () => skewSetter(proxy.skew) });
        }
    }
});


document$.subscribe(() => {
    let scrollTimeout;

    const triggerHighlights = () => {
        // Sirf un 'strong' tags ko pakdo jo screen par dikh rahe hain
        const highlights = document.querySelectorAll('.md-typeset strong:not(.highlight-active)');

        highlights.forEach((el, index) => {
            const rect = el.getBoundingClientRect();
            const isVisible = rect.top >= 0 && rect.bottom <= window.innerHeight;

            if (isVisible) {
                // Thoda sa delay (stagger) taaki ek saath saare na khulein
                setTimeout(() => {
                    el.classList.add('highlight-active');
                    // Optional: Pencil scribble sound yahan play kar sakte ho
                    if (typeof playSFX === 'function') playSFX('pencil-short', 0.2);
                }, index * 150);
            }
        });
    };

    // Scroll event listener
    window.addEventListener('scroll', () => {
        // Jab tak scroll ho raha hai, purana timeout clear karo
        clearTimeout(scrollTimeout);

        // Jab scroll ruke (250ms tak koi movement na ho)
        scrollTimeout = setTimeout(() => {
            triggerHighlights();
        }, 250);
    });

    // Initial check agar page load par hi kuch dikh raha ho
    setTimeout(triggerHighlights, 1000);
});

// Function to handle the slide-out before navigation
const handleNavigation = (e) => {
    const link = e.target.closest("a");
    if (!link) return;

    const href = link.getAttribute("href");
    const target = link.getAttribute("target");

    // Skip if: no href, internal anchor, external link, or opens in new tab
    if (!href || href.startsWith("#") || target === "_blank" || href.includes("http") && !href.includes(window.location.hostname)) {
        return;
    }

    e.preventDefault();
    e.stopPropagation();

    const main = document.querySelector(".md-main__inner");

    // 1. EXIT ANIMATION: Purana page niche slide hoga
    gsap.to(main, {
        y: "100vh",
        opacity: 0,
        duration: 0.6,
        ease: "power2.in",
        onStart: () => {
            document.body.style.overflow = "hidden";
        },
        onComplete: () => {
            // Animation khatam hone par hi page change karo
            sessionStorage.setItem("rolling", "true");
            location.href = href;
        }
    });
};

// 2. ENTRANCE ANIMATION: Jab naya page load ho
document$.subscribe(() => {
    const main = document.querySelector(".md-main__inner");
    if (!main) return;

    // Click listeners ko reset karna (Memory leak se bachne ke liye)
    document.removeEventListener("click", handleNavigation);
    document.addEventListener("click", handleNavigation);

    if (sessionStorage.getItem("rolling") === "true") {
        sessionStorage.removeItem("rolling");

        // Set initial state (Upar chhupa hua)
        gsap.set(main, { y: "-100vh", opacity: 0 });

        // Slide In Animation
        gsap.to(main, {
            y: 0,
            opacity: 1,
            duration: 0.8,
            ease: "back.out(1.2)", // Halka sa bounce effect
            onComplete: () => {
                document.body.style.overflow = "";
                gsap.set(main, { clearProps: "all" });
            }
        });
    }
});



// --- CONFIGURATION ---
const slideDuration = 0.7;
const paperEase = "power4.inOut";

// 1. EXIT ANIMATION (Purana panna niche jaana)
const exitPage = (container) => {
    return gsap.to(container, {
        y: "100vh",
        opacity: 0,
        duration: slideDuration,
        ease: "power2.in",
        display: "none" // Taaki naya content turant peeche hide na ho
    });
};

// 2. ENTRANCE ANIMATION (Naya panna upar se aana)
const enterPage = (container) => {
    // Initial position set karo (Screen ke upar)
    gsap.set(container, {
        y: "-100vh",
        opacity: 0,
        display: "block"
    });

    // Niche slide karke center mein laao
    return gsap.to(container, {
        y: 0,
        opacity: 1,
        duration: slideDuration + 0.2,
        ease: "back.out(1.1)", // Halka sa bouncy landing
        onStart: () => {
            if (window.playSFX) playSFX('paper-slide-in', 0.4);
        },
        onComplete: () => {
            gsap.set(container, { clearProps: "y,opacity" });
            document.body.style.overflow = ""; // Scroll wapas enable
        }
    });
};
