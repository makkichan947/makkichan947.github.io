// Makki Theme 主JavaScript文件

document.addEventListener('DOMContentLoaded', function() {
    // 初始化主题
    initTheme();
    
    // 初始化搜索功能
    initSearch();
    
    // 初始化动画效果
    initAnimations();
    
    // 初始化代码高亮
    initCodeHighlight();
});

// 主题初始化
function initTheme() {
    // 添加淡入动画类
    const elements = document.querySelectorAll('.content-wrapper, .card, .post-article');
    elements.forEach((el, index) => {
        setTimeout(() => {
            el.classList.add('fade-in-up');
        }, index * 100);
    });
}

// 搜索功能初始化
function initSearch() {
    const searchToggle = document.getElementById('search-toggle');
    const searchModal = document.getElementById('search-modal');
    const searchInput = document.getElementById('search-input');
    const mainSearch = document.getElementById('main-search');
    
    // 搜索切换按钮
    if (searchToggle && searchModal) {
        searchToggle.addEventListener('click', function() {
            searchModal.style.display = searchModal.style.display === 'block' ? 'none' : 'block';
            if (searchModal.style.display === 'block') {
                searchInput.focus();
            }
        });
    }
    
    // 点击模态框外部关闭
    if (searchModal) {
        searchModal.addEventListener('click', function(e) {
            if (e.target === searchModal) {
                searchModal.style.display = 'none';
            }
        });
    }
    
    // ESC键关闭搜索
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && searchModal) {
            searchModal.style.display = 'none';
        }
        if ((e.key === '/' || e.key === 'k' && (e.ctrlKey || e.metaKey)) && searchInput) {
            e.preventDefault();
            if (searchModal) {
                searchModal.style.display = 'block';
                searchInput.focus();
            } else if (mainSearch) {
                mainSearch.focus();
            }
        }
    });
    
    // 搜索输入处理
    [searchInput, mainSearch].forEach(input => {
        if (input) {
            input.addEventListener('input', function(e) {
                const query = e.target.value.trim();
                if (query.length > 2) {
                    performSearch(query);
                } else {
                    hideSearchResults();
                }
            });
        }
    });
}

// 执行搜索
function performSearch(query) {
    // 这里实现搜索逻辑
    console.log('搜索:', query);
    // 实际项目中需要集成lunr.js或其他搜索库
}

// 隐藏搜索结果
function hideSearchResults() {
    const results = document.getElementById('search-results');
    if (results) {
        results.innerHTML = '';
    }
}

// 初始化动画效果
function initAnimations() {
    // 滚动触发动画
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, observerOptions);
    
    // 观察元素
    const animateElements = document.querySelectorAll('.card, .post-article, .wiki-article');
    animateElements.forEach(el => {
        observer.observe(el);
    });
}

// 初始化代码高亮
function initCodeHighlight() {
    // 渲染数学公式
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\[', right: '\\]', display: true},
                {left: '\\(', right: '\\)', display: false}
            ]
        });
    }
}

// 工具函数
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 复制代码功能
function initCopyCode() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        // 添加复制按钮
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.textContent = '复制';
        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(block.textContent);
            copyButton.textContent = '已复制!';
            setTimeout(() => {
                copyButton.textContent = '复制';
            }, 2000);
        });
        
        const pre = block.parentElement;
        pre.style.position = 'relative';
        pre.appendChild(copyButton);
    });
}

// 平滑滚动到顶部
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// 主题切换功能（预留）
function toggleTheme() {
    // 这里可以实现明暗主题切换
    console.log('主题切换功能预留');
}

// 导出函数供模板使用
window.MakkiTheme = {
    scrollToTop,
    toggleTheme,
    performSearch,
    debounce
};