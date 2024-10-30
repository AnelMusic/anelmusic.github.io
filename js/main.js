// File: js/main.js
// Highlight current section in TOC
const observerOptions = {
    root: null,
    rootMargin: '-80px 0px -50% 0px',
    threshold: 0
};

const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            document.querySelectorAll('.toc a').forEach(link => {
                link.classList.remove('active');
            });
            
            const id = entry.target.getAttribute('id');
            const correspondingLink = document.querySelector(`.toc a[href="#${id}"]`);
            if (correspondingLink) {
                correspondingLink.classList.add('active');
            }
        }
    });
}, observerOptions);

// Initialize observers for the current page
function initializeObservers() {
    const sections = document.querySelectorAll('.about-section, .category-section');
    sections.forEach(section => {
        observer.observe(section);
    });
}

// Smooth scroll for TOC links
function initializeSmoothScroll() {
    document.querySelectorAll('.toc a, .sidebar a').forEach(link => {
        link.addEventListener('click', (e) => {
            // Only handle internal links
            if (link.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const targetId = link.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeObservers();
    initializeSmoothScroll();
});
