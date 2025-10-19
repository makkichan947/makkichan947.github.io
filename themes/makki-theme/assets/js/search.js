// 搜索功能实现
// 使用Lunr.js进行全文搜索

let searchIndex = null;
let searchData = null;

// 初始化搜索索引
function initSearchIndex() {
    // 获取搜索数据
    fetch('/index.json')
        .then(response => response.json())
        .then(data => {
            searchData = data;
            
            // 创建Lunr索引
            searchIndex = lunr(function() {
                this.ref('uri');
                this.field('title', { boost: 10 });
                this.field('content', { boost: 5 });
                this.field('tags', { boost: 8 });
                this.field('categories');
                
                data.forEach(page => {
                    this.add({
                        uri: page.uri,
                        title: page.title,
                        content: page.content,
                        tags: page.tags,
                        categories: page.categories
                    });
                });
            });
        })
        .catch(error => {
            console.error('搜索索引加载失败:', error);
        });
}

// 执行搜索
function performSearch(query) {
    if (!searchIndex || !query) return;
    
    const results = searchIndex.search(query);
    displaySearchResults(results, query);
}

// 显示搜索结果
function displaySearchResults(results, query) {
    const searchResults = document.getElementById('search-results');
    if (!searchResults) return;
    
    if (results.length === 0) {
        searchResults.innerHTML = `
            <div class="search-no-results">
                <p>没有找到与 "${query}" 相关的内容</p>
                <p>尝试使用不同的关键词搜索</p>
            </div>
        `;
        return;
    }
    
    const resultsHTML = results.slice(0, 10).map(result => {
        const page = searchData.find(p => p.uri === result.ref);
        if (!page) return '';
        
        return `
            <div class="search-result">
                <h3><a href="${page.uri}">${page.title}</a></h3>
                <p>${getSnippet(page.content, query)}</p>
                <div class="search-meta">
                    <span class="search-type">${page.section}</span>
                    ${page.tags ? page.tags.map(tag => `<span class="search-tag">${tag}</span>`).join('') : ''}
                </div>
            </div>
        `;
    }).join('');
    
    searchResults.innerHTML = `<div class="search-results-container">${resultsHTML}</div>`;
}

// 获取搜索片段
function getSnippet(content, query) {
    if (!content) return '';
    
    const words = query.split(' ').filter(word => word.length > 0);
    let snippet = content.substring(0, 200);
    
    // 尝试在片段中包含搜索词
    for (const word of words) {
        const index = content.toLowerCase().indexOf(word.toLowerCase());
        if (index !== -1 && index < 200) {
            snippet = content.substring(Math.max(0, index - 50), Math.min(content.length, index + 150));
            break;
        }
    }
    
    return snippet + (content.length > 200 ? '...' : '');
}

// 绑定搜索事件
document.addEventListener('DOMContentLoaded', function() {
    initSearchIndex();
    
    // 绑定搜索输入框事件
    const searchInputs = document.querySelectorAll('#search-input, #main-search');
    searchInputs.forEach(input => {
        if (input) {
            // 使用防抖优化搜索性能
            const debouncedSearch = window.MakkiTheme.debounce(function(e) {
                const query = e.target.value.trim();
                if (query.length > 2) {
                    performSearch(query);
                } else {
                    hideSearchResults();
                }
            }, 300);
            
            input.addEventListener('input', debouncedSearch);
        }
    });
});

// 隐藏搜索结果
function hideSearchResults() {
    const searchResults = document.getElementById('search-results');
    if (searchResults) {
        searchResults.innerHTML = '';
    }
}

// 键盘快捷键支持
document.addEventListener('keydown', function(e) {
    // Ctrl+K 或 Cmd+K 聚焦搜索
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('search-input') || document.getElementById('main-search');
        if (searchInput) {
            searchInput.focus();
        }
    }
});

// 导出搜索函数
window.SearchModule = {
    initSearchIndex,
    performSearch,
    displaySearchResults,
    hideSearchResults
};