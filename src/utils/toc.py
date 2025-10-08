"""
Table of Contents (TOC) module for automatic sidebar navigation.
Provides wrapper functions that automatically register sections for the TOC.
"""

import streamlit as st
import re

# Global list to collect sections during page render
_current_page_sections = []


def generate_anchor(title):
    """
    Generate a URL-friendly anchor from a title.
    
    Args:
        title (str): The section title
        
    Returns:
        str: URL-friendly anchor
    """
    # Remove emojis and special characters
    clean_title = re.sub(r'[^\w\s-]', '', title)
    # Convert to lowercase and replace spaces with hyphens
    anchor = clean_title.lower().strip().replace(' ', '-')
    # Remove multiple consecutive hyphens
    anchor = re.sub(r'-+', '-', anchor)
    return anchor


def init_toc():
    """Initialize the TOC for the current page render."""
    global _current_page_sections
    _current_page_sections = []


def _register_section(title, level):
    """
    Register a section in the TOC.
    
    Args:
        title (str): The section title
        level (int): The heading level (1 for header, 2 for subheader, etc.)
    """
    global _current_page_sections
    
    anchor = generate_anchor(title)
    _current_page_sections.append({
        'title': title,
        'level': level,
        'anchor': anchor
    })


def toc_header(title, **kwargs):
    """
    Wrapper for st.header() that automatically registers the section in TOC.
    
    Args:
        title (str): The header title
        **kwargs: Additional arguments passed to st.header()
    """
    anchor = generate_anchor(title)
    _register_section(title, level=1)
    st.header(title, anchor=anchor, **kwargs)


def toc_subheader(title, **kwargs):
    """
    Wrapper for st.subheader() that automatically registers the section in TOC.
    
    Args:
        title (str): The subheader title
        **kwargs: Additional arguments passed to st.subheader()
    """
    anchor = generate_anchor(title)
    _register_section(title, level=2)
    st.subheader(title, anchor=anchor, **kwargs)


def toc_subsubheader(title, **kwargs):
    """
    Wrapper for level 3 heading that automatically registers the section in TOC.
    
    Args:
        title (str): The subsubheader title
        **kwargs: Additional arguments passed to st.markdown()
    """
    anchor = generate_anchor(title)
    _register_section(title, level=3)
    st.markdown(f"### {title}", **kwargs)


def toc_markdown(html, level=1, **kwargs):
    """
    Wrapper for st.markdown() that automatically registers HTML headers in TOC.
    
    Args:
        html (str): The HTML markdown string
        level (int): The heading level (1 for h2, 2 for h3, etc.)
        **kwargs: Additional arguments passed to st.markdown()
    """
    # Extract title from HTML (remove tags and emojis for TOC)
    title_match = re.search(r'<h\d[^>]*>(.*?)</h\d>', html)
    if title_match:
        title = title_match.group(1)
        # Remove HTML tags from title
        clean_title = re.sub(r'<[^>]+>', '', title)
        
        anchor = generate_anchor(clean_title)
        _register_section(clean_title, level=level)
        
        # Add anchor to the HTML if not already present
        if 'id=' not in html:
            # Insert id attribute properly within the opening tag
            html = re.sub(r'<(h\d)(\s+[^>]*)?>', rf'<\1 id="{anchor}"\2>', html)
    
    st.markdown(html, **kwargs)


def _generate_numbering(sections):
    """
    Generate hierarchical numbering for sections.
    
    Args:
        sections (list): List of section dictionaries
        
    Returns:
        list: List of section numbers (e.g., "1", "1.1", "1.2", "2", "2.1")
    """
    numbers = []
    counters = [0, 0, 0]  # Support up to 3 levels
    
    for section in sections:
        level = section['level']
        
        if level == 1:
            counters[0] += 1
            counters[1] = 0
            counters[2] = 0
            numbers.append(f"{counters[0]}")
        elif level == 2:
            counters[1] += 1
            counters[2] = 0
            numbers.append(f"{counters[0]}.{counters[1]}")
        elif level == 3:
            counters[2] += 1
            numbers.append(f"{counters[0]}.{counters[1]}.{counters[2]}")
        else:
            numbers.append("")
    
    return numbers


def render_toc():
    """
    Render the Table of Contents in the sidebar.
    Must be called at the END of the page after all sections are registered.
    """
    global _current_page_sections
    
    if not _current_page_sections:
        return
    
    sections = _current_page_sections
    numbers = _generate_numbering(sections)
    
    # Build TOC content
    toc_content = "---\n\n### 📑 Table of Contents\n\n"
    
    for i, section in enumerate(sections):
        title = section['title']
        level = section['level']
        anchor = section['anchor']
        number = numbers[i]
        
        # Calculate indentation
        indent = "&nbsp;" * 4 * (level - 1)
        
        # Create the link
        toc_content += f"{indent}{number}. [{title}](#{anchor})\n\n"
    
    # Display in sidebar
    st.sidebar.markdown(toc_content, unsafe_allow_html=True)

