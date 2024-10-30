/* File: css/styles.css */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: #333;
    display: flex;
    min-height: 100vh;
    position: relative;
}

.navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: #1a1a1a;
    padding: 1rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    z-index: 1000;
    height: 60px;
}

.navbar a {
    color: #fff;
    text-decoration: none;
}

.content-wrapper {
    display: flex;
    margin-top: 60px;
    width: 100%;
}

.sidebar {
    width: 250px;
    background: #f5f5f5;
    padding: 2rem;
    position: fixed;
    left: 0;
    top: 60px;
    bottom: 0;
    overflow-y: auto;
}

.main-content {
    margin-left: 250px;
    margin-right: 300px;
    padding: 2rem;
    max-width: 800px;
}

.right-sidebar {
    width: 300px;
    padding: 2rem 1.5rem;
    position: fixed;
    right: 0;
    top: 60px;
    bottom: 0;
    background: #ffffff;
    border-left: 1px solid #eee;
    overflow-y: auto;
}

.category-section {
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #eee;
}

.category-section:last-child {
    border-bottom: none;
}

.category-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.category-header h2 {
    margin-bottom: 0;
}

.view-all {
    color: #4169e1;
    text-decoration: none;
    font-size: 0.9rem;
}

.view-all:hover {
    text-decoration: underline;
}

.category-description {
    color: #666;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
}

h1, h2, h3 {
    margin-bottom: 1.5rem;
    color: #333;
    scroll-margin-top: 80px;
}

p {
    margin-bottom: 1.5rem;
}

.contact-links {
    margin: 2rem 0;
    padding: 1.5rem;
    background: #f8f9fa;
    border-radius: 8px;
}

.contact-links h3 {
    margin-bottom: 1rem;
}

.contact-links ul {
    list-style: none;
}

.contact-links li {
    margin-bottom: 0.75rem;
}

.contact-links a {
    color: #4169e1;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.contact-links a:hover {
    text-decoration: underline;
}

.expertise-list {
    list-style: none;
    margin: 1.5rem 0;
}

.expertise-list li {
    margin-bottom: 1rem;
    padding-left: 1.5rem;
    position: relative;
}

.expertise-list li::before {
    content: "•";
    color: #4169e1;
    font-weight: bold;
    position: absolute;
    left: 0;
}

.toc {
    font-size: 0.9rem;
}

.toc h2 {
    font-size: 1rem;
    color: #666;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.toc ul {
    list-style: none;
}

.toc li {
    margin-bottom: 1rem;
}

.toc a {
    color: #666;
    text-decoration: none;
    display: block;
    padding: 0.2rem 0;
    transition: color 0.2s;
}

.toc a:hover {
    color: #4169e1;
}

.toc a.active {
    color: #4169e1;
    font-weight: 500;
}

.category-list {
    list-style: none;
    margin-bottom: 2rem;
}

.category-list h3 {
    font-size: 1rem;
    color: #666;
    margin-bottom: 0.5rem;
}

.category-list a {
    color: #555;
    text-decoration: none;
    display: block;
    padding: 0.3rem 0;
    font-size: 0.9rem;
}

.category-list a:hover {
    color: #4169e1;
}

.social-button {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #4169e1;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    margin: 1rem 0;
}

.article-list {
    list-style: none;
}

.article-list li {
    margin-bottom: 1rem;
}

.article-list a {
    color: #4169e1;
    text-decoration: none;
    display: block;
}

.article-list a:hover {
    text-decoration: underline;
}

.article-date {
    color: #666;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

.about-section {
    margin-bottom: 3rem;
}

/* Responsive styles */
@media (max-width: 1200px) {
    .right-sidebar {
        display: none;
    }
    .main-content {
        margin-right: 0;
    }
}

@media (max-width: 768px) {
    .sidebar {
        display: none;
    }
    .main-content {
        margin-left: 0;
    }
}
