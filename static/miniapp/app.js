document.body.classList.add('app-booting');
const tg = window.Telegram?.WebApp || {
            expand() {}, ready() {}, close() {}, openLink(url) { window.open(url, '_blank'); },
            BackButton: { show() {}, hide() {}, onClick() {}, offClick() {} },
            HapticFeedback: { notificationOccurred() {}, impactOccurred() {} }, initDataUnsafe: {}, initData: ''
        };
        tg.expand();
        if (
            typeof tg.requestFullscreen === 'function'
            && (typeof tg.isVersionAtLeast !== 'function' || tg.isVersionAtLeast('8.0'))
        ) {
            try {
                const fullscreenRequest = tg.requestFullscreen();
                if (fullscreenRequest && typeof fullscreenRequest.catch === 'function') {
                    fullscreenRequest.catch(() => {});
                }
            } catch (error) {
                console.debug('Telegram fullscreen is not available in this client:', error);
            }
        }
        tg.ready();
        const BOT_USERNAME = "FlimfyBoxBot"; // change if needed

        // State
        let allMovies = [];
        let tmdbMoviesMap = {};
        let activeMovie = null;
        let activeDetailsMovieId = null;
        const movieDetailsCache = new Map();
        let savedMovieIds = new Set();
        let myListToggleInFlight = false;
        let myListRequestId = 0;
        let detailsRequestId = 0;
        let heroTimer = null;
        let heroItems = [];
        let heroIndex = 0;
        let newReleaseRequestId = 0;
        let trendingRequestId = 0;
        let genreCatalog = null;
        let genreCatalogType = null;
        let genreCatalogRequestId = 0;
        let genreDetailRequestId = 0;
        let genreDetailMovies = [];
        let genreDetailVisibleCount = 24;
        let genreDetailActiveId = null;
        let browseType = 'all';
        let browseRequestId = 0;
        let globalChatTimer = null;
        let detailsController = null;
        let searchController = null;
        let browseController = null;
        let genreController = null;
        let initialHomePending = 0;
        let initialHomeTimeout = null;
        let telegramBackButtonHandler = null;
        const HOME_CATALOGUE_CACHE_KEY = 'flimfybox-home-catalogue-v2';
        const HOME_CATALOGUE_CACHE_TTL = 10 * 60 * 1000;

        function parseMiniAppEpisodeMetadata(value) {
            const text = String(value || '').toUpperCase();
            const seasonMatch = text.match(/\bS(?:EASON)?\s*0*(\d+)\b/);
            const season = seasonMatch ? parseInt(seasonMatch[1], 10) : null;
            const rangeMatch = text.match(
                /\b(?:S\s*0*\d+\s*)?(?:E(?:PISODE)?|EP)\s*0*(\d+)\s*(?:[-~]\s*(?:E(?:PISODE)?|EP)?\s*0*(\d+))\b/
            );
            const singleMatch = rangeMatch ? null : text.match(
                /\b(?:S\s*0*\d+\s*)?(?:E(?:PISODE)?|EP)\s*0*(\d+)\b/
            );
            const episodeStart = rangeMatch
                ? parseInt(rangeMatch[1], 10)
                : (singleMatch ? parseInt(singleMatch[1], 10) : null);
            const episodeEnd = rangeMatch ? parseInt(rangeMatch[2], 10) : null;
            return { season, episodeStart, episodeEnd };
        }

        function formatMiniAppQuality(value) {
            return String(value || '')
                .replace(/\([^)]*\)/g, ' ')
                .replace(/\[[^\]]*(?:GB|MB|KB)\s*\]/gi, ' ')
                .replace(/\bS(?:EASON)?\s*0*\d+\s*(?:E(?:PISODE)?|EP)?\s*0*\d*(?:\s*[-~]\s*(?:E(?:PISODE)?|EP)?\s*0*\d+)?\b/gi, ' ')
                .replace(/\b(?:E(?:PISODE)?|EP)\s*0*\d+(?:\s*[-~]\s*(?:E(?:PISODE)?|EP)?\s*0*\d+)?\b/gi, ' ')
                .replace(/\b(?:COMPLETE|COMBINED|COMBIND)\b/gi, ' ')
                .replace(/\s+/g, ' ')
                .trim();
        }

        function syncTelegramBackButton() {
            const backButton = tg.BackButton;
            if (!backButton || typeof backButton.show !== 'function') return;
            if (telegramBackButtonHandler && typeof backButton.offClick === 'function') {
                backButton.offClick(telegramBackButtonHandler);
            }
            telegramBackButtonHandler = () => {
                const detailsPage = document.getElementById('detailsPage');
                const genreDetailPage = document.getElementById('genreDetailPage');
                if (detailsPage?.classList.contains('open')) {
                    window.closeDetails();
                } else if (genreDetailPage?.classList.contains('open')) {
                    window.closeGenreDetail();
                } else if (!document.body.classList.contains('screen-home')) {
                    window.showHome();
                } else {
                    tg.close();
                }
            };
            if (typeof backButton.onClick === 'function') backButton.onClick(telegramBackButtonHandler);
            backButton.show();
        }

        function readHomeCatalogueCache() {
            try {
                const cached = JSON.parse(localStorage.getItem(HOME_CATALOGUE_CACHE_KEY) || 'null');
                if (!cached || !Array.isArray(cached.movies) || !cached.savedAt) return null;
                return {
                    movies: cached.movies,
                    hasMore: cached.hasMore !== false,
                    fresh: Date.now() - cached.savedAt < HOME_CATALOGUE_CACHE_TTL
                };
            } catch (_error) {
                localStorage.removeItem(HOME_CATALOGUE_CACHE_KEY);
                return null;
            }
        }

        function writeHomeCatalogueCache(movies, hasMore) {
            try {
                localStorage.setItem(HOME_CATALOGUE_CACHE_KEY, JSON.stringify({
                    movies,
                    hasMore: hasMore !== false,
                    savedAt: Date.now()
                }));
            } catch (error) {
                console.warn('Home catalogue cache could not be saved:', error);
            }
        }

        async function loadHomeSection(url, cacheKey) {
            const storageKey = `flimfybox-home-section-${cacheKey}-v3`;
            let cached = null;
            try {
                cached = JSON.parse(localStorage.getItem(storageKey) || 'null');
                if (cached && cached.savedAt && cached.data) {
                    const isFresh = Date.now() - cached.savedAt < HOME_CATALOGUE_CACHE_TTL;
                    if (!isFresh) {
                        // Stale-while-refresh: never hide a previously loaded
                        // Home section just because its refresh window expired.
                        refreshHomeSection(url, cacheKey, storageKey);
                    }
                    return cached.data;
                }
            } catch (_error) {
                localStorage.removeItem(storageKey);
            }

            return fetchHomeSection(url, cacheKey, storageKey);
        }

        async function refreshHomeSection(url, cacheKey, storageKey) {
            try {
                await fetchHomeSection(url, cacheKey, storageKey);
            } catch (error) {
                console.warn(`Home section refresh failed (${cacheKey}):`, error);
            }
        }

        async function fetchHomeSection(url, cacheKey, storageKey) {
            const response = await fetch(url);
            const data = await response.json();
            if (!response.ok || data.status === 'error') {
                throw new Error(data.message || 'Could not load home section');
            }
            try {
                localStorage.setItem(storageKey, JSON.stringify({ data, savedAt: Date.now() }));
            } catch (error) {
                console.warn(`Home section cache could not be saved (${cacheKey}):`, error);
            }
            return data;
        }

        function startInitialHomeLoading() {
            // Never block the Home shell behind a full-page loader. Cached
            // catalogue data renders immediately; network refreshes continue
            // in the background.
            initialHomePending = 0;
            document.body.classList.remove('app-booting');
            const screen = document.getElementById('appLoadingScreen');
            if (screen) screen.classList.add('is-complete');
            if (initialHomeTimeout) clearTimeout(initialHomeTimeout);
            initialHomeTimeout = null;
        }

        function completeInitialHomeLoadingStep() {
            if (initialHomePending <= 0) return;
            initialHomePending -= 1;
            if (initialHomePending > 0) return;
            if (initialHomeTimeout) {
                clearTimeout(initialHomeTimeout);
                initialHomeTimeout = null;
            }
            document.body.classList.remove('app-booting');
            const screen = document.getElementById('appLoadingScreen');
            if (screen) screen.classList.add('is-complete');
        }

        const CANONICAL_GENRE_TAXONOMY = [
            { id: 'action', label: 'Action', group: 'Action & Adventure', description: 'Fast, explosive, and high-stakes viewing.' },
            { id: 'adventure', label: 'Adventure', group: 'Action & Adventure', description: 'Quest-driven stories and faraway journeys.' },
            { id: 'animation', label: 'Animation', group: 'Animation & Anime', description: 'Bold visual storytelling and inventive worlds.' },
            { id: 'anime', label: 'Anime', group: 'Animation & Anime', description: 'Anime, manga-inspired worlds and beloved fandoms.' },
            { id: 'biography', label: 'Biography', group: 'Drama & Documentary', description: 'Real-life stories with emotional depth and context.' },
            { id: 'comedy', label: 'Comedy', group: 'Comedy', description: 'Light, witty, and crowd-pleasing entertainment.' },
            { id: 'crime', label: 'Crime', group: 'Crime & Mystery', description: 'Sharp investigations, pressure, and power plays.' },
            { id: 'documentary', label: 'Documentary', group: 'Documentary & Reality', description: 'Truth, culture, and world-expanding nonfiction.' },
            { id: 'drama', label: 'Drama', group: 'Drama', description: 'Character-first stories with emotional weight.' },
            { id: 'erotic', label: 'Erotic', group: 'Adult / Mature', description: 'Mature, romantic, and explicit content.' },
            { id: 'family', label: 'Family', group: 'Family & Feel-Good', description: 'Warm and accessible storytelling for every age.' },
            { id: 'fantasy', label: 'Fantasy', group: 'Sci‑Fi & Fantasy', description: 'Magic, myth, and imaginative wonder.' },
            { id: 'history', label: 'History', group: 'Drama & Documentary', description: 'Historic eras, legacies, and major turning points.' },
            { id: 'horror', label: 'Horror', group: 'Horror', description: 'Suspense, dread, and fear-driven cinema.' },
            { id: 'music', label: 'Music', group: 'Music & Culture', description: 'Performance, rhythm, and creative energy.' },
            { id: 'mystery', label: 'Mystery', group: 'Crime & Mystery', description: 'Clues, tension, and cinematic intrigue.' },
            { id: 'reality', label: 'Reality', group: 'Documentary & Reality', description: 'Authentic, unscripted, and high-interest stories.' },
            { id: 'romance', label: 'Romance', group: 'Romance', description: 'Emotional connection, chemistry, and longing.' },
            { id: 'scifi', label: 'Sci‑Fi', group: 'Sci‑Fi & Fantasy', description: 'Future worlds, tech, and speculative fiction.' },
            { id: 'short', label: 'Short', group: 'Short-form', description: 'Compact, sharp, and instantly watchable stories.' },
            { id: 'sport', label: 'Sport', group: 'Sport & Lifestyle', description: 'Competition, ambition, and underdog energy.' },
            { id: 'thriller', label: 'Thriller', group: 'Thriller', description: 'Tension, danger, and edge-of-the-seat suspense.' },
            { id: 'war', label: 'War', group: 'Action & Adventure', description: 'Conflict, endurance, and high-stakes battlefields.' },
            { id: 'western', label: 'Western', group: 'Classic & Cinematic', description: 'Dust, law, and frontier storytelling.' },
            { id: 'musical', label: 'Musical', group: 'Music & Culture', description: 'Songs, emotion, and performance-led stories.' },
            { id: 'cyberpunk', label: 'Cyberpunk', group: 'Sci‑Fi & Fantasy', description: 'Neon futures, corporate tension, and dystopian energy.' },
            { id: 'dystopian', label: 'Dystopian', group: 'Sci‑Fi & Fantasy', description: 'Oppressive worlds and fragile resistance.' },
            { id: 'space-opera', label: 'Space Opera', group: 'Sci‑Fi & Fantasy', description: 'Epic cosmic scale and heroic journeys.' },
            { id: 'psychological-horror', label: 'Psychological Horror', group: 'Horror', description: 'Fear driven by the mind, guilt, and dread.' },
            { id: 'supernatural-horror', label: 'Supernatural Horror', group: 'Horror', description: 'Hauntings, curses, and unknown forces.' },
            { id: 'political-drama', label: 'Political Drama', group: 'Drama', description: 'Power, consequence, and high-pressure choices.' },
            { id: 'historical-drama', label: 'Historical Drama', group: 'Drama', description: 'Large-scale narratives shaped by eras and events.' },
            { id: 'romantic-comedy', label: 'Romantic Comedy', group: 'Comedy', description: 'Warm chemistry and easygoing fun.' },
            { id: 'dark-comedy', label: 'Dark Comedy', group: 'Comedy', description: 'Humor built on chaos, irony, and edge.' },
            { id: 'detective', label: 'Detective', group: 'Crime & Mystery', description: 'Clues, suspects, and moral puzzles.' },
            { id: 'procedural', label: 'Procedural', group: 'Crime & Mystery', description: 'Case-by-case tension and investigative structure.' },
            { id: 'heist', label: 'Heist', group: 'Action & Adventure', description: 'Strategy, stealing, and adrenaline.' },
            { id: 'survival', label: 'Survival', group: 'Action & Adventure', description: 'Danger, resilience, and impossible odds.' },
            { id: 'superhero', label: 'Superhero', group: 'Action & Adventure', description: 'Legend, power, and larger-than-life conflict.' },
            { id: 'spy', label: 'Spy', group: 'Action & Adventure', description: 'Espionage, stealth, and geopolitical thrill.' },
            { id: 'anime-genre', label: 'Anime', group: 'Animation & Anime', description: 'Animated storytelling with massive cultural reach.' },
            { id: 'isekai', label: 'Isekai', group: 'Animation & Anime', description: 'Reborn into a new world and adventure unfolds.' },
            { id: 'shonen', label: 'Shonen', group: 'Animation & Anime', description: 'High energy, ambition, and action-driven arcs.' },
            { id: 'shojo', label: 'Shojo', group: 'Animation & Anime', description: 'Romance, emotion, and beautifully layered character arcs.' },
            { id: 'seinen', label: 'Seinen', group: 'Animation & Anime', description: 'Mature tone and layered themes.' },
            { id: 'kaiju', label: 'Kaiju', group: 'Animation & Anime', description: 'Monster-scale spectacle and destruction.' },
            { id: 'zombie', label: 'Zombie', group: 'Horror', description: 'Apocalypse, panic, and survival instincts.' },
            { id: 'vampire', label: 'Vampire', group: 'Horror', description: 'Dark romance, bloodlust, and nightfall.' }
        ];

        const CANONICAL_GENRE_MAP = Object.fromEntries(CANONICAL_GENRE_TAXONOMY.map(genre => [genre.id, genre]));
        const GENRE_GROUP_ORDER = ['Action & Adventure', 'Crime & Mystery', 'Sci‑Fi & Fantasy', 'Horror', 'Drama', 'Comedy', 'Animation & Anime', 'Documentary & Reality', 'Music & Culture', 'Romance', 'Family & Feel-Good', 'Sport & Lifestyle', 'Short-form', 'Classic & Cinematic', 'Adult / Mature'];
        const MOBILE_FEATURED_GENRE_ORDER = ['action', 'comedy', 'drama', 'horror', 'animation', 'thriller', 'romance', 'scifi', 'crime', 'fantasy'];
        const GENRE_VISUAL_PALETTE = {
            'Action & Adventure': { from: '#f97316', to: '#7c2d12', glow: 'rgba(249, 115, 22, .38)' },
            'Crime & Mystery': { from: '#38bdf8', to: '#0f172a', glow: 'rgba(59, 130, 246, .28)' },
            'Sci‑Fi & Fantasy': { from: '#8b5cf6', to: '#0f172a', glow: 'rgba(139, 92, 246, .35)' },
            'Horror': { from: '#ef4444', to: '#111827', glow: 'rgba(239, 68, 68, .35)' },
            'Drama': { from: '#f59e0b', to: '#3f3f46', glow: 'rgba(245, 158, 11, .25)' },
            'Comedy': { from: '#f472b6', to: '#4c1d95', glow: 'rgba(244, 114, 182, .28)' },
            'Animation & Anime': { from: '#22c55e', to: '#0f766e', glow: 'rgba(34, 197, 94, .24)' },
            'Documentary & Reality': { from: '#a78bfa', to: '#1e293b', glow: 'rgba(167, 139, 250, .24)' },
            'Music & Culture': { from: '#fb7185', to: '#312e81', glow: 'rgba(251, 113, 133, .24)' },
            'Romance': { from: '#f472b6', to: '#7c2d12', glow: 'rgba(244, 114, 182, .25)' },
            'Family & Feel-Good': { from: '#34d399', to: '#14532d', glow: 'rgba(52, 211, 153, .22)' },
            'Sport & Lifestyle': { from: '#14b8a6', to: '#0f172a', glow: 'rgba(20, 184, 166, .24)' },
            'Short-form': { from: '#38bdf8', to: '#1d4ed8', glow: 'rgba(56, 189, 248, .22)' },
            'Classic & Cinematic': { from: '#c084fc', to: '#312e81', glow: 'rgba(192, 132, 252, .24)' },
            'Adult / Mature': { from: '#f87171', to: '#3f3f46', glow: 'rgba(248, 113, 113, .2)' }
        };
        const POSTER_PLACEHOLDER = '/static/miniapp/poster-placeholder.svg';
        const IMAGE_FALLBACK_GRADIENT = 'linear-gradient(135deg, #1b2037, #080a12)';
        // Stable curated poster assignment. These fixed, recognizable movie
        // key-art URLs never read catalogue movies, counts, or API results.
        const GENRE_ARTWORK = {
            action: 'https://upload.wikimedia.org/wikipedia/en/6/6e/Mad_Max_Fury_Road.jpg',
            adventure: 'https://upload.wikimedia.org/wikipedia/en/e/e7/Jurassic_Park_poster.jpg',
            animation: 'https://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg',
            anime: 'https://upload.wikimedia.org/wikipedia/en/d/db/Spirited_Away_Japanese_poster.png',
            'anime-genre': 'https://m.media-amazon.com/images/M/MV5BMjI1ODZkYTgtYTY3Yy00ZTJkLWFkOTgtZDUyYWM4MzQwNjk0XkEyXkFqcGc@._V1_.jpg',
            biography: 'https://upload.wikimedia.org/wikipedia/en/9/9f/Bohemian_Rhapsody.png',
            comedy: 'https://upload.wikimedia.org/wikipedia/en/b/b9/Hangoverposter09.jpg',
            crime: 'https://upload.wikimedia.org/wikipedia/en/1/1c/Godfather_ver1.jpg',
            documentary: 'https://upload.wikimedia.org/wikipedia/en/9/9c/Free_Solo.png',
            drama: 'https://upload.wikimedia.org/wikipedia/en/b/b8/A_Beautiful_Mind_Poster.jpg',
            erotic: 'https://upload.wikimedia.org/wikipedia/en/5/5e/50ShadesofGreyCoverArt.jpg',
            family: 'https://m.media-amazon.com/images/M/MV5BMTAxOTMwOTkwNDZeQTJeQWpwZ15BbWU4MDEyMTI1NjMx._V1_.jpg',
            fantasy: 'https://upload.wikimedia.org/wikipedia/en/f/fb/Lord_Rings_Fellowship_Ring.jpg',
            history: 'https://upload.wikimedia.org/wikipedia/en/3/38/Schindler%27s_List_movie.jpg',
            horror: 'https://upload.wikimedia.org/wikipedia/en/7/7b/Exorcist_ver2.jpg',
            music: 'https://m.media-amazon.com/images/M/MV5BMDFjOWFkYzktYzhhMC00NmYyLTkwY2EtYjViMDhmNzg0OGFkXkEyXkFqcGc@._V1_.jpg',
            mystery: 'https://upload.wikimedia.org/wikipedia/en/1/1f/Knives_Out_poster.jpeg',
            reality: 'https://upload.wikimedia.org/wikipedia/en/c/cd/Trumanshow.jpg',
            romance: 'https://upload.wikimedia.org/wikipedia/en/8/86/Posternotebook.jpg',
            scifi: 'https://upload.wikimedia.org/wikipedia/en/b/b4/Spider-Man-_Across_the_Spider-Verse_poster.jpg',
            short: 'https://upload.wikimedia.org/wikipedia/en/8/89/Le_ballon_rouge_%281956%29.png',
            sport: 'https://upload.wikimedia.org/wikipedia/en/1/18/Rocky_poster.jpg',
            thriller: 'https://m.media-amazon.com/images/M/MV5BMTk0MDQ3MzAzOV5BMl5BanBnXkFtZTgwNzU1NzE3MjE@._V1_.jpg',
            war: 'https://m.media-amazon.com/images/M/MV5BYzkxZjg2NDQtMGVjMy00NWZkLTk0ZDEtZWE3NDYwYjAyMTg1XkEyXkFqcGc@._V1_.jpg',
            western: 'https://upload.wikimedia.org/wikipedia/en/4/45/Good_the_bad_and_the_ugly_poster.jpg',
            musical: 'https://m.media-amazon.com/images/M/MV5BMDllYjliOTUtMDJjZC00ODIzLWJmNGMtOWI2NzQxMjA2NzdlXkEyXkFqcGc@._V1_.jpg',
            cyberpunk: 'https://m.media-amazon.com/images/M/MV5BOWQ4YTBmNTQtMDYxMC00NGFjLTkwOGQtNzdhNmY1Nzc1MzUxXkEyXkFqcGc@._V1_.jpg',
            dystopian: 'https://m.media-amazon.com/images/M/MV5BMWI1OGM4YjQtNmIxNi00YmE2LWJkNTAtY2Q0YjU4NTI5NWQyXkEyXkFqcGc@._V1_.jpg',
            'space-opera': 'https://m.media-amazon.com/images/M/MV5BYjRkYzAzNjktZmRhMy00NjRiLWE0OTMtYmRmMTE5NDkzY2NlXkEyXkFqcGc@._V1_.jpg',
            'psychological-horror': 'https://m.media-amazon.com/images/M/MV5BNzY2NzI4OTE5MF5BMl5BanBnXkFtZTcwMjMyNDY4Mw@@._V1_.jpg',
            'supernatural-horror': 'https://m.media-amazon.com/images/M/MV5BMTM3NjA1NDMyMV5BMl5BanBnXkFtZTcwMDQzNDMzOQ@@._V1_.jpg',
            'political-drama': 'https://m.media-amazon.com/images/M/MV5BZGQzMzcwMDYtMmNjNS00YzZlLTg2MjUtNTE0MThlNTFjMDQ0XkEyXkFqcGc@._V1_.jpg',
            'historical-drama': 'https://m.media-amazon.com/images/M/MV5BYWQ4YmNjYjEtOWE1Zi00Y2U4LWI4NTAtMTU0MjkxNWQ1ZmJiXkEyXkFqcGc@._V1_.jpg',
            'romantic-comedy': 'https://m.media-amazon.com/images/M/MV5BMTYxNDMyOTAxN15BMl5BanBnXkFtZTgwMDg1ODYzNTM@._V1_.jpg',
            'dark-comedy': 'https://m.media-amazon.com/images/M/MV5BMzM5NjUxOTEyMl5BMl5BanBnXkFtZTgwNjEyMDM0MDE@._V1_.jpg',
            detective: 'https://m.media-amazon.com/images/M/MV5BMTg0NjEwNjUxM15BMl5BanBnXkFtZTcwMzk0MjQ5Mg@@._V1_.jpg',
            procedural: 'https://m.media-amazon.com/images/M/MV5BNDFkMTRkZmQtM2I0NC00NjJjLWJlMDctNTNiZWYxYzhjZDZiXkEyXkFqcGc@._V1_.jpg',
            heist: 'https://m.media-amazon.com/images/M/MV5BMmNhZDkxYTgtMDM3ZC00NTQ3LWFjZTUtNzc1Y2QyNWZjNDRmXkEyXkFqcGc@._V1_.jpg',
            survival: 'https://m.media-amazon.com/images/M/MV5BYTgwNmQzZDctMjNmOS00OTExLTkwM2UtNzJmOTJhODFjOTdlXkEyXkFqcGc@._V1_.jpg',
            superhero: 'https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg',
            spy: 'https://m.media-amazon.com/images/M/MV5BMWQ1ZDM4NDktMWY0NC00MjcxLWJlMDMtNmE2MGVhYzRjMWQ0XkEyXkFqcGc@._V1_.jpg',
            isekai: 'https://m.media-amazon.com/images/M/MV5BN2NhYzU2NDEtYzI1NS00MjgzLThjZGUtOTYxNGJkZjZmNDdjXkEyXkFqcGc@._V1_.jpg',
            shonen: 'https://m.media-amazon.com/images/M/MV5BMTA5MTc1M2EtZWQ2Ni00ZmU2LTg3MzQtOTliMjE4OGM0ZWFiXkEyXkFqcGc@._V1_.jpg',
            shojo: 'https://m.media-amazon.com/images/M/MV5BZTEyZDhlNDctMGMyNy00YTczLTgyMjktNGRjMDI2MzM0YTU5XkEyXkFqcGc@._V1_.jpg',
            seinen: 'https://m.media-amazon.com/images/M/MV5BNzljMjA3MTQtMjM1OS00OGJjLWJiYzctZDRiMTk1NWI5YzQ5XkEyXkFqcGc@._V1_.jpg',
            kaiju: 'https://m.media-amazon.com/images/M/MV5BODE2NTdmMmYtY2U1OS00MjExLWIwNjQtYjQ5NTA0ZDZmZjZiXkEyXkFqcGc@._V1_.jpg',
            zombie: 'https://m.media-amazon.com/images/M/MV5BODA3OTM4NWQtMjU1OS00NzA2LTlhZDAtYjU2YjM5MWUxNTdiXkEyXkFqcGc@._V1_.jpg',
            vampire: 'https://m.media-amazon.com/images/M/MV5BNjY4NDlkMzctMzRmZC00YWZjLTg3MGItZTI2M2NkZjg3YTIxXkEyXkFqcGc@._V1_.jpg'
        };

        function getGenreVisualPalette(groupName) {
            return GENRE_VISUAL_PALETTE[groupName] || { from: '#8b7bff', to: '#0f172a', glow: 'rgba(139, 123, 255, .24)' };
        }

        const GENRE_ALIASES = {
            action: ['action', 'martial arts', 'martial-arts', 'superhero', 'spy', 'heist', 'survival', 'war action', 'war-action', 'battle'],
            adventure: ['adventure', 'quest', 'expedition', 'journey'],
            animation: ['animation', 'animated', 'cartoons', 'cartoon'],
            anime: ['anime', 'anime genre', 'isekai', 'shonen', 'shojo', 'seinen', 'kaiju'],
            biography: ['biography', 'bio'],
            comedy: ['comedy', 'dark comedy', 'romantic comedy', 'sitcom', 'sketch comedy', 'action comedy', 'comdey', 'conedy', 'omedy', 'comedy drama', 'comedy. drama'],
            crime: ['crime', 'cime', 'gangster', 'detective', 'procedural', 'heist'],
            documentary: ['documentary', 'docudrama', 'real life', 'reality documentary'],
            drama: ['drama', 'dram', 'drame', 'family drama', 'political drama', 'medical drama', 'legal drama', 'historical drama', 'coming-of-age', 'teen drama'],
            erotic: ['erotic', 'adult', 'hot', 'unrated', '18+'],
            family: ['family', 'kids', 'children'],
            fantasy: ['fantasy', 'myth', 'dark fantasy'],
            history: ['history', 'histry', 'historical', 'period'],
            horror: ['horror', 'psychological horror', 'supernatural horror', 'slasher', 'monster', 'zombie', 'vampire', 'haunting'],
            music: ['music', 'musical'],
            mystery: ['mystery', 'investigation', 'whodunit', 'detective'],
            reality: ['reality', 'reality show', 'reality-tv', 'reality tv', 'tv shows', 'talk show', 'game show', 'talk-show', 'game-show', 'sitcom', 'reality-series'],
            romance: ['romance', 'romantic', 'love story', 'love-story'],
            scifi: ['sci-fi', 'sci fi', 'science fiction', 'science-fiction', 'cyberpunk', 'dystopian', 'time travel', 'space opera', 'mecha'],
            short: ['short', 'short film'],
            sport: ['sport', 'sports', 'wrestling'],
            thriller: ['thriller', 'suspense', 'psychological', 'crime thriller'],
            war: ['war', 'war action thriller'],
            western: ['western', 'frontier'],
            musical: ['musical', 'song'],
            cyberpunk: ['cyberpunk'],
            dystopian: ['dystopian'],
            'space-opera': ['space opera', 'space-opera'],
            'psychological-horror': ['psychological horror', 'psychological-horror'],
            'supernatural-horror': ['supernatural horror', 'supernatural-horror'],
            'political-drama': ['political drama', 'political-drama'],
            'historical-drama': ['historical drama', 'historical-drama'],
            'romantic-comedy': ['romantic comedy', 'romantic-comedy'],
            'dark-comedy': ['dark comedy', 'dark-comedy'],
            detective: ['detective'],
            procedural: ['procedural'],
            heist: ['heist'],
            survival: ['survival'],
            superhero: ['superhero'],
            spy: ['spy', 'espionage'],
            'isekai': ['isekai'],
            shonen: ['shonen'],
            shojo: ['shojo'],
            seinen: ['seinen'],
            kaiju: ['kaiju'],
            zombie: ['zombie'],
            vampire: ['vampire']
        };

        const KNOWN_GENRE_GARBAGE = new Set([
            '', 'n/a', 'na', 'a', 'n', 'unknown', 'quality : bluray', 'quality : hdcam', 'quality : hdrip',
            'quality : hdtc', 'quality : hdts', 'quality : web-dl', 'bluray', 'hdcam', 'hdrip', 'hdtc', 'hdts',
            'web-dl', 'dubbed', 'random', 'kids', 'adult', 'hot', 'unrated', 'political', 'social', 'devotional',
            'survival', 'spy', 'turkish', 'john putch', 'simran choudhary', 'nandu vijay krishna', 'stars: chammak chandra',
            'cime', 'comdey', 'conedy', 'omedy', 'com�dieetthriller', 'reality tv', 'reality-tv', 'talk-show', 'talk show', 'game-show',
            'game show', 'tv-shows', 'tv shows', 'tv reality show', 'reality'
        ]);

        function normalizeGenreText(rawText) {
            if (!rawText || typeof rawText !== 'string') return [];
            const tokens = rawText.split(/[,&/|]/).map(part => part.trim()).filter(Boolean);
            const canonical = new Set();
            tokens.forEach(token => {
                let value = token
                    .replace(/[–—]/g, '-')
                    .replace(/[\u2018\u2019]/g, "'")
                    .replace(/\s+/g, ' ')
                    .trim()
                    .toLowerCase();
                value = value.replace(/\.+/g, ' ').replace(/[^a-z0-9\s-]/g, ' ');
                value = value.replace(/\s+/g, ' ').trim();
                if (!value || KNOWN_GENRE_GARBAGE.has(value)) return;
                const matched = Object.keys(GENRE_ALIASES).find(key => GENRE_ALIASES[key].includes(value));
                if (matched) {
                    canonical.add(matched);
                    return;
                }
                if (value.includes('science fiction') || value.includes('sci fi')) {
                    canonical.add('scifi');
                    return;
                }
                if (value.includes('dark comedy')) {
                    canonical.add('dark-comedy');
                    return;
                }
                if (value.includes('political drama')) {
                    canonical.add('political-drama');
                    return;
                }
                if (value.includes('historical drama')) {
                    canonical.add('historical-drama');
                    return;
                }
                if (value.includes('romantic comedy')) {
                    canonical.add('romantic-comedy');
                    return;
                }
                if (value.includes('supernatural horror')) {
                    canonical.add('supernatural-horror');
                    return;
                }
                if (value.includes('psychological horror')) {
                    canonical.add('psychological-horror');
                    return;
                }
                if (value.includes('space opera')) {
                    canonical.add('space-opera');
                    return;
                }
                if (value.includes('cyberpunk')) {
                    canonical.add('cyberpunk');
                }
            });
            return Array.from(canonical);
        }

        function findGenreById(genreId) {
            return CANONICAL_GENRE_TAXONOMY.find(genre => genre.id === genreId) || CANONICAL_GENRE_TAXONOMY[0];
        }

        async function loadGenreCatalog(force = false) {
            const requestedType = browseType;
            if (genreCatalog && genreCatalogType === requestedType && !force) return genreCatalog;
            const requestId = ++genreCatalogRequestId;
            const response = await fetch(`/api/genres?type=${encodeURIComponent(requestedType)}`);
            const data = await response.json();
            if (!response.ok || data.status !== 'success') throw new Error(data.message || 'Could not load genres');
            if (requestId !== genreCatalogRequestId || requestedType !== browseType) return null;
            genreCatalog = Object.fromEntries((data.genres || []).map(genre => [genre.id, genre]));
            genreCatalogType = requestedType;
            return genreCatalog;
        }

        async function loadBrowseCollections() {
            const requestId = ++browseRequestId;
            if (browseController) browseController.abort();
            browseController = new AbortController();
            const signal = browseController.signal;
            const container = document.getElementById('browseCollections');
            container.innerHTML = '<div class="genre-loading"><div class="loader"></div><span>Curating decade collections…</span></div>';
            try {
                const data = await apiRequest(`/api/browse?type=${encodeURIComponent(browseType)}`, { signal });
                if (requestId !== browseRequestId) return;
                const collectionCaption = browseType === 'tv'
                    ? 'Web series & TV shows'
                    : browseType === 'movies'
                        ? 'Movies only'
                        : 'Movies, series & TV';
                container.innerHTML = (data.collections || []).filter(collection => collection.movies.length).map(collection => `
                    <section class="browse-collection movie-row">
                        <div class="row-header"><div class="row-header-left">${collection.label}</div><span class="row-caption">${collectionCaption}</span></div>
                        <div class="horizontal-scroll">${renderCards(collection.movies, 'card', false)}</div>
                    </section>
                `).join('');
            } catch (error) {
                if (isAbortError(error)) return;
                if (requestId === browseRequestId) container.innerHTML = `<div class="empty-search-state">${error.message}</div>`;
            }
        }

        window.setBrowseType = async function(type) {
            if (!['all', 'movies', 'tv'].includes(type) || type === browseType && genreCatalogType === type) return;
            browseType = type;
            genreCatalog = null;
            genreCatalogType = null;
            genreCatalogRequestId++;
            genreDetailVisibleCount = 24;
            document.querySelectorAll('[data-browse-type]').forEach(button => {
                button.classList.toggle('active', button.dataset.browseType === browseType);
            });
            document.getElementById('browseCollections').innerHTML = '';
            renderExploreScreen();
            await loadBrowseCollections();
            if (document.getElementById('genreDetailPage')?.classList.contains('open') && genreDetailActiveId) {
                await window.openGenreDetail(genreDetailActiveId);
            }
        };

        window.surpriseMe = async function() {
            const result = document.getElementById('surpriseResult');
            const recommendation = document.getElementById('surpriseRecommendation');
            recommendation.hidden = false;
            result.innerHTML = '<div class="surprise-loading">Finding your next watch…</div>';
            try {
                const response = await fetch(`/api/browse/surprise?type=${encodeURIComponent(browseType)}`);
                const data = await response.json();
                if (!response.ok || data.status !== 'success') throw new Error(data.message || 'Could not find a title');
                if (!data.movie) {
                    result.innerHTML = `<div class="surprise-empty">${data.message || 'No eligible local titles are available.'}</div>`;
                    return;
                }
                const movie = data.movie;
                trackRecommendationEvent('surprise_impression', movie.id, { browse_type: browseType });
                if (!allMovies.some(item => String(item.id) === String(movie.id))) allMovies.push(movie);
                result.innerHTML = renderCards([movie], 'grid-card', false);
                const card = result.querySelector('.grid-card, .card');
                if (card) {
                    card.addEventListener('click', () => {
                        trackRecommendationEvent('surprise_click', movie.id, { browse_type: browseType });
                    }, { once: true });
                }
            } catch (error) {
                result.innerHTML = `<div class="surprise-empty">${error.message || 'Could not find a title'}</div>`;
            }
        };

        // Utility
        function showToast(msg) {
            const t = document.getElementById('toast');
            t.innerText = msg;
            t.classList.add('show');
            setTimeout(() => t.classList.remove('show'), 2500);
        }

        function scrollRow(elementId, amount) {
            const el = document.getElementById(elementId);
            if (el) el.scrollBy({ left: amount, behavior: 'smooth' });
        }

        function telegramAuthHeaders() {
            return tg.initData ? { 'X-Telegram-Init-Data': tg.initData } : {};
        }

        function isAbortError(error) {
            return error && error.name === 'AbortError';
        }

        async function apiRequest(url, options = {}) {
            const response = await fetch(url, options);
            let data;
            try {
                data = await response.json();
            } catch (_error) {
                throw new Error(response.ok ? 'The server returned an invalid response.' : 'The server is temporarily unavailable.');
            }
            if (!response.ok || (data && data.status === 'error')) {
                throw new Error(data?.message || 'The request could not be completed.');
            }
            return data;
        }

        function guestChatToken() {
            const storageKey = 'flimfybox-chat-guest-token';
            let token = localStorage.getItem(storageKey);
            if (!token) {
                token = `${crypto.randomUUID()}-${crypto.randomUUID()}`;
                localStorage.setItem(storageKey, token);
            }
            return token;
        }

        function globalChatHeaders() {
            return {
                ...telegramAuthHeaders(),
                'X-Guest-Token': guestChatToken()
            };
        }

        async function refreshMyListButton(movieId, button) {
            if (!tg.initData || !button) return;
            try {
                const response = await fetch(`/api/my-list/${encodeURIComponent(movieId)}/status`, {
                    headers: telegramAuthHeaders()
                });
                const data = await response.json();
                if (!response.ok || data.status !== 'success') return;
                if (String(activeMovie?.id) !== String(movieId)) return;
                if (data.saved) {
                    savedMovieIds.add(String(movieId));
                    button.innerHTML = '<i class="fas fa-check"></i>';
                } else {
                    savedMovieIds.delete(String(movieId));
                    button.innerHTML = '<i class="fas fa-plus"></i>';
                }
            } catch (error) {
                console.warn('Could not refresh My List state:', error);
            }
        }

        function trackRecommendationEvent(eventType, movieId = null, metadata = {}) {
            if (!tg.initData) return;
            fetch('/api/recommendation-events', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...telegramAuthHeaders() },
                body: JSON.stringify({ event_type: eventType, movie_id: movieId, metadata })
            }).catch(error => console.warn('Recommendation event was not recorded:', error));
        }

        function setActiveNav(index) {
            document.querySelectorAll('.nav-item').forEach((item, itemIndex) => {
                item.classList.toggle('active', itemIndex === index);
            });
        }

        window.openMorePanel = function() {
            document.getElementById('morePanel').classList.add('open');
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
        };

        window.closeMorePanel = function() {
            document.getElementById('morePanel').classList.remove('open');
        };

        window.closeInfoModal = function() {
            document.getElementById('infoModal').classList.remove('open');
            document.getElementById('infoModal').classList.remove('chat-modal');
            if (globalChatTimer) {
                clearInterval(globalChatTimer);
                globalChatTimer = null;
            }
        };

        function setupMotionEffects() {
            if (!('IntersectionObserver' in window)) return;
            const observer = new IntersectionObserver(entries => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('is-revealed');
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.12, rootMargin: '0px 0px -30px' });
            const observeRows = () => document.querySelectorAll('.movie-row:not(.is-revealed)').forEach(row => {
                observer.observe(row);
                row.querySelectorAll('.card').forEach(card => card.classList.add('is-revealed'));
            });
            observeRows();
            new MutationObserver(observeRows).observe(document.getElementById('mainContent'), { childList: true, subtree: true });
        }

        window.showInfoPanel = function(title, body, icon = 'circle-info', actions = '') {
            closeMorePanel();
            document.getElementById('infoModalTitle').textContent = title;
            document.getElementById('infoModalBody').innerHTML = body;
            document.getElementById('infoModalIcon').innerHTML = `<i class="fas fa-${icon}"></i>`;
            document.getElementById('infoModalActions').innerHTML = actions || '<button type="button" class="primary-action" onclick="closeInfoModal()">Done</button>';
            document.getElementById('infoModal').classList.add('open');
        };

        window.showTvSeries = function() {
            closeMorePanel();
            showExplore();
            window.setBrowseType('tv');
        };

        window.showUpcoming = function() {
            closeMorePanel();
            showHome();
            const row = document.getElementById('rowUpcoming');
            if (row) {
                row.scrollIntoView({ behavior: 'smooth', block: 'center' });
                document.getElementById('upcomingScroll')?.focus({ preventScroll: true });
            }
        };

        window.showGlobalChat = function() {
            closeMorePanel();
            if (globalChatTimer) clearInterval(globalChatTimer);
            showInfoPanel('Global Chat', `
                <div class="chat-shell">
                    <div class="chat-room-banner">
                        <div class="chat-room-icon"><i class="fas fa-earth-americas"></i></div>
                        <div><strong>FlimfyBox Community</strong><span>Everyone can join the conversation</span></div>
                        <span class="chat-live"><i></i> Live</span>
                    </div>
                    <div class="chat-messages" aria-live="polite"><div class="chat-loading"><i class="fas fa-spinner fa-spin"></i><span>Loading the community…</span></div></div>
                    <form class="chat-form" onsubmit="sendGlobalChat(event)">
                        <input class="chat-input" maxlength="500" autocomplete="off" placeholder="Share something with the community…" required>
                        <button class="chat-send" type="submit" aria-label="Send message"><i class="fas fa-paper-plane"></i></button>
                    </form>
                    <div class="chat-hint"><i class="fas fa-shield-halved"></i> Keep it friendly · 500 characters max</div>
                </div>
            `, 'comment', '');
            document.getElementById('infoModal').classList.add('chat-modal');
            loadGlobalChat();
            globalChatTimer = setInterval(loadGlobalChat, 5000);
        };

        window.loadGlobalChat = function() {
            fetch('/api/global-chat', { headers: globalChatHeaders() })
                .then(response => response.json().then(data => ({ ok: response.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok) throw new Error(data.message || 'Could not load Global Chat.');
                    const messages = document.querySelector('.chat-messages');
                    if (!messages) return;
                    messages.innerHTML = (data.messages || []).map(item => {
                        const author = item.username ? `@${item.username}` : (item.first_name || 'Community member');
                        const initials = escapeHtml(author.replace(/^@/, '').slice(0, 1).toUpperCase());
                        const time = item.created_at ? new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
                        return `<article class="chat-message">
                            <div class="chat-avatar">${initials}</div>
                            <div class="chat-message-body">
                                <div class="chat-message-meta"><strong>${escapeHtml(author)}</strong><time>${escapeHtml(time)}</time></div>
                                <div class="chat-text">${escapeHtml(item.message)}</div>
                            </div>
                        </article>`;
                    }).join('') || '<div class="chat-empty"><i class="fas fa-comments"></i><strong>No messages yet</strong><span>Start the conversation with the community.</span></div>';
                    messages.scrollTop = messages.scrollHeight;
                })
                .catch(error => {
                    const messages = document.querySelector('.chat-messages');
                    if (messages) messages.innerHTML = `<div class="chat-empty chat-error"><i class="fas fa-cloud-exclamation"></i><strong>Chat is taking a break</strong><span>${escapeHtml(error.message)}</span></div>`;
                });
        };

        window.sendGlobalChat = function(event) {
            event.preventDefault();
            const form = event.currentTarget;
            const input = form.querySelector('.chat-input');
            const message = input.value.trim();
            if (!message) return;
            input.disabled = true;
            fetch('/api/global-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...globalChatHeaders() },
                body: JSON.stringify({ message })
            })
                .then(response => response.json().then(data => ({ ok: response.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok) throw new Error(data.message || 'Could not send message');
                    input.value = '';
                    loadGlobalChat();
                })
                .catch(error => showToast(error.message))
                .finally(() => { input.disabled = false; });
        };

        function escapeHtml(value) {
            return String(value || '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
        }

        window.showSettings = function() {
            const notifications = localStorage.getItem('flimfybox-notifications') !== 'off';
            const compact = localStorage.getItem('flimfybox-compact') === 'on';
            showInfoPanel('Settings', `
                <label class="settings-toggle"><span>New title notifications</span><input id="settingsNotifications" type="checkbox" ${notifications ? 'checked' : ''} onchange="localStorage.setItem('flimfybox-notifications', this.checked ? 'on' : 'off'); showToast(this.checked ? 'Notifications enabled' : 'Notifications disabled')"></label>
                <label class="settings-toggle"><span>Compact cards</span><input id="settingsCompact" type="checkbox" ${compact ? 'checked' : ''} onchange="localStorage.setItem('flimfybox-compact', this.checked ? 'on' : 'off'); document.body.classList.toggle('compact-cards', this.checked); showToast('Display preference saved')"></label>
                <p class="info-note">Your preferences are saved on this device.</p>
            `, 'gear', '<button type="button" class="primary-action" onclick="closeInfoModal()">Save & close</button>');
        };

        document.addEventListener('keydown', event => {
            if (event.key === 'Escape') closeInfoModal();
        });
        document.body.classList.toggle('compact-cards', localStorage.getItem('flimfybox-compact') === 'on');

        window.showHome = function() {
            closeMorePanel();
            closeInfoModal();
            document.body.classList.remove('screen-explore', 'screen-search', 'screen-list');
            document.body.classList.add('screen-home');
            document.getElementById('mainContent').style.display = '';
            document.getElementById('myListContent').style.display = 'none';
            document.getElementById('searchResultsContent').style.display = 'none';
            document.getElementById('exploreContent').style.display = 'none';
            document.getElementById('genreDetailPage').classList.remove('open');
            document.querySelectorAll('.movie-row').forEach(row => row.style.display = '');
            setActiveNav(0);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        };

        window.showExplore = function() {
            closeMorePanel();
            closeInfoModal();
            document.body.classList.remove('screen-home', 'screen-search', 'screen-list');
            document.body.classList.add('screen-explore');
            document.getElementById('mainContent').style.display = 'none';
            document.getElementById('myListContent').style.display = 'none';
            document.getElementById('searchResultsContent').style.display = 'none';
            document.getElementById('exploreContent').style.display = 'block';
            document.getElementById('genreDetailPage').classList.remove('open');
            document.getElementById('searchDropdown').classList.remove('active');
            setActiveNav(2);
            renderExploreScreen();
            loadBrowseCollections();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        };

        window.showSearch = function() {
            closeMorePanel();
            closeInfoModal();
            showHome();
            document.body.classList.remove('screen-home', 'screen-explore', 'screen-list');
            document.body.classList.add('screen-search');
            setActiveNav(1);
            document.getElementById('mainContent').style.display = 'none';
            document.getElementById('exploreContent').style.display = 'none';
            document.getElementById('myListContent').style.display = 'none';
            document.getElementById('searchResultsContent').style.display = 'block';
            renderSearchDiscovery();
            setTimeout(() => document.getElementById('searchInput').focus(), 120);
        };

        function renderSearchDiscovery() {
            const dropdown = document.getElementById('searchDropdown');
            const recent = getRecentSearches();
            dropdown.innerHTML = recent.length
                ? `<div class="search-empty-state"><strong>Recent searches</strong><div class="search-recent">${recent.map(item => `
                    <button class="search-recent-chip" type="button" data-recent-search="${escapeHtml(item)}">
                        <i class="fas fa-clock-rotate-left"></i>${escapeHtml(item)}
                    </button>`).join('')}</div></div>`
                : '<div class="search-empty-state"><strong>Search your catalogue</strong><span>Try a title, genre, or keyword.</span></div>';
            dropdown.querySelectorAll('[data-recent-search]').forEach(button => {
                button.addEventListener('click', () => {
                    const input = document.getElementById('searchInput');
                    input.value = button.dataset.recentSearch || '';
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                });
            });
            dropdown.classList.add('active');
        }

        window.showMyList = async function() {
            closeMorePanel();
            closeInfoModal();
            document.body.classList.remove('screen-home', 'screen-explore', 'screen-search');
            document.body.classList.add('screen-list');
            const requestId = ++myListRequestId;
            const grid = document.getElementById('myListGrid');
            document.getElementById('mainContent').style.display = 'none';
            document.getElementById('searchResultsContent').style.display = 'none';
            document.getElementById('exploreContent').style.display = 'none';
            document.getElementById('genreDetailPage').classList.remove('open');
            document.getElementById('myListContent').style.display = 'block';
            document.getElementById('searchDropdown').classList.remove('active');
            setActiveNav(3);
            grid.innerHTML = '<div class="loader" style="grid-column:1/-1">Loading your saved titles…</div>';
            try {
                const response = await fetch('/api/my-list', { headers: telegramAuthHeaders() });
                const data = await response.json();
                if (requestId !== myListRequestId) return;
                if (!response.ok || data.status !== 'success') throw new Error(data.message || 'Could not load My List');
                (data.movies || []).forEach(movie => {
                    if (!allMovies.some(existing => String(existing.id) === String(movie.id))) allMovies.push(movie);
                });
                savedMovieIds = new Set((data.movies || []).map(movie => String(movie.id)));
                grid.innerHTML = data.movies?.length
                    ? renderCards(data.movies, 'grid-card', false)
                    : '<div class="empty-my-list">Your list is empty.<br><span>Save a title with the + button.</span></div>';
            } catch (error) {
                if (requestId !== myListRequestId) return;
                grid.innerHTML = `<div class="empty-my-list">${error.message}</div>`;
            }
        };

        window.toggleCurrentMyList = async function() {
            const movieId = activeDetailsMovieId;
            const mutationMovieId = String(movieId || '');
            if (!movieId || !activeMovie || activeMovie.source === 'tmdb' || String(movieId).startsWith('tmdb_')) {
                showToast('Only available titles can be saved right now.');
                return;
            }
            if (myListToggleInFlight) return;
            myListToggleInFlight = true;
            const button = document.getElementById('detailMyListButton');
            if (button) button.disabled = true;
            try {
                // Snapshot the details-page movie. The home carousel changes in
                // the background, so never read a mutable global after await.
                const statusResponse = await fetch(`/api/my-list/${encodeURIComponent(mutationMovieId)}/status`, {
                    headers: telegramAuthHeaders()
                });
                const statusData = await statusResponse.json();
                if (!statusResponse.ok || statusData.status !== 'success') {
                    throw new Error(statusData.message || 'Could not check My List');
                }
                const isSaved = Boolean(statusData.saved);
                if (isSaved) savedMovieIds.add(movieId);
                else savedMovieIds.delete(movieId);
                const response = await fetch(isSaved ? `/api/my-list/${movieId}` : '/api/my-list', {
                    method: isSaved ? 'DELETE' : 'POST',
                    headers: { 'Content-Type': 'application/json', ...telegramAuthHeaders() },
                    body: JSON.stringify({ movie_id: movieId })
                });
                const data = await response.json();
                if (!response.ok || data.status !== 'success') throw new Error(data.message || 'Could not save title');
                if (String(data.movie_id) !== mutationMovieId || String(activeDetailsMovieId) !== mutationMovieId) {
                    throw new Error('The selected title changed. Please try again.');
                }
                const saved = Boolean(data.saved);
                if (!saved) {
                    savedMovieIds.delete(movieId);
                    if (button) button.innerHTML = '<i class="fas fa-plus"></i>';
                    trackRecommendationEvent('watchlist_remove', movieId);
                    showToast('Removed from My List');
                } else {
                    savedMovieIds.add(movieId);
                    if (button) button.innerHTML = '<i class="fas fa-check"></i>';
                    trackRecommendationEvent('watchlist_add', movieId);
                    showToast('Saved to My List');
                }
            } catch (error) {
                showToast(error.message || 'Could not save title');
            } finally {
                myListToggleInFlight = false;
                if (button) button.disabled = false;
            }
        };

        // Pagination State
        let currentPage = 1;
        let isFetching = false;
        let hasMoreMovies = true;

        // Load movies from API with Infinite Scroll support
        async function loadMovies(page = 1) {
            if (isFetching || !hasMoreMovies) return;
            isFetching = true;
            const cachedHome = page === 1 ? readHomeCatalogueCache() : null;

            try {
                if (cachedHome) {
                    allMovies = cachedHome.movies;
                    hasMoreMovies = cachedHome.hasMore;
                    renderHome(allMovies);
                    currentPage = 2;
                    completeInitialHomeLoadingStep();
                    if (cachedHome.fresh) {
                        isFetching = false;
                        return;
                    }
                }

                // Agar page 1 se zyada hai, toh neeche ek loading spinner dikhao
                if (page > 1) {
                    document.getElementById('moreGrid').insertAdjacentHTML('beforeend', '<div id="scrollLoader" style="grid-column: 1 / -1; text-align: center; padding: 20px;"><div class="loader" style="width:30px;height:30px;border-width:3px;margin:0 auto;"></div></div>');
                }

                const data = await apiRequest(`/api/movies?page=${page}&limit=40`);
                
                // Naya data aate hi loader hata do
                if (page > 1) {
                    const loader = document.getElementById('scrollLoader');
                    if (loader) loader.remove();
                }

                if (data.status === 'success') {
                    const newMovies = data.movies.filter(m => m.image);
                    hasMoreMovies = data.has_more; 
                    
                    if (page === 1) {
                        // Pehli baar: Pura UI setup karo
                        allMovies = newMovies;
                        writeHomeCatalogueCache(newMovies, data.has_more);
                        renderHome(allMovies); 
                        completeInitialHomeLoadingStep();
                    } else {
                        // Scrolling par: Purani movies mein nayi jod do
                        allMovies = [...allMovies, ...newMovies]; 
                        const newCardsHTML = renderCards(newMovies, 'grid-card', false);
                        document.getElementById('moreGrid').insertAdjacentHTML('beforeend', newCardsHTML);
                    }
                    currentPage++; // Agli baar ke liye page badha do
                }
            } catch (e) {
                if (isAbortError(e)) return;
                console.error('Fetch failed', e);
                document.getElementById('moreGrid').innerHTML = '<div class="empty-search-state" style="grid-column:1/-1">We could not load the catalogue.<br><span>Check your connection and try again.</span></div>';
                if (page === 1) completeInitialHomeLoadingStep();
            } finally {
                isFetching = false;
            }
        }

        // 🔥 NAYA: Infinite Scroll Listener
        window.addEventListener('scroll', () => {
            // Agar user page ke bottom se 600px upar hai, toh advance mein next page load kar lo
            if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 600) {
                // Check karo ki normal page open hai (Search result open na ho)
                if (document.getElementById('searchResultsContent').style.display === 'none') {
                    loadMovies(currentPage);
                }
            }
        });

        function renderExploreScreen() {
            if (!genreCatalog) {
                document.getElementById('exploreBody').innerHTML = '<div class="genre-loading"><div class="loader"></div><span>Mapping the real catalogue…</span></div>';
                loadGenreCatalog().then(() => {
                    if (genreCatalog && genreCatalogType === browseType) renderExploreScreen();
                }).catch(error => {
                    document.getElementById('exploreBody').innerHTML = `<div class="empty-search-state">${error.message}</div>`;
                });
                return;
            }

            const query = (document.getElementById('genreSearchInput')?.value || '').trim().toLowerCase();
            const visibleGenres = CANONICAL_GENRE_TAXONOMY.filter(genre => {
                const catalogueGenre = genreCatalog[genre.id];
                // A selected media type should never show empty genre tiles.
                if (browseType !== 'all' && (!catalogueGenre || catalogueGenre.count < 1)) return false;
                const match = genre.label.toLowerCase().includes(query) || genre.group.toLowerCase().includes(query) || genre.description.toLowerCase().includes(query);
                const subqueryMatch = Object.keys(GENRE_ALIASES).some(aliasKey => {
                    const aliasList = GENRE_ALIASES[aliasKey] || [];
                    return aliasList.some(alias => alias.toLowerCase().includes(query));
                });
                return !query || match || subqueryMatch;
            });

            const isCompactBrowse = window.matchMedia('(max-width: 699px)').matches;
            const orderedGenres = isCompactBrowse
                ? visibleGenres
                    .filter(genre => MOBILE_FEATURED_GENRE_ORDER.includes(genre.id))
                    .sort((a, b) => MOBILE_FEATURED_GENRE_ORDER.indexOf(a.id) - MOBILE_FEATURED_GENRE_ORDER.indexOf(b.id))
                : visibleGenres;
            const grouped = isCompactBrowse
                ? [{ groupName: '', items: orderedGenres }]
                : GENRE_GROUP_ORDER.map(groupName => {
                const items = visibleGenres.filter(genre => genre.group === groupName);
                return { groupName, items };
            }).filter(section => section.items.length);

            const body = document.getElementById('exploreBody');
            if (!grouped.length) {
                body.innerHTML = '<div class="empty-search-state" style="margin:20px;">No genres match your search.</div>';
                return;
            }

            body.innerHTML = grouped.map(section => {
                const cards = section.items.map(genre => {
                    const catalogueGenre = genreCatalog[genre.id] || {};
                    const count = catalogueGenre.count || 0;
                    // Browse tiles use curated, stable artwork. Catalogue posters
                    // belong to the genre detail results and must not replace these.
                    const posters = [GENRE_ARTWORK[genre.id]];
                    const palette = getGenreVisualPalette(genre.group);
                    return `
                        <button class="genre-card genre-${genre.id} ${count === 0 ? 'empty' : ''}" type="button" data-genre-group="${genre.group}" onclick="openGenreDetail('${genre.id}')" style="--genre-from: ${palette.from}; --genre-to: ${palette.to}; --genre-glow: ${palette.glow};">
                            <span class="genre-card-posters poster-count-${Math.min(posters.length, 3)}" aria-hidden="true">
                                ${posters.slice(0, 3).map((poster, index) => `<img class="genre-card-poster genre-card-poster-${index + 1}" src="${poster}" alt="" loading="lazy" decoding="async">`).join('')}
                            </span>
                            <div class="genre-card-meta">
                                <span>${count} ${count === 1 ? 'title' : 'titles'}</span>
                                <span>${genre.group}</span>
                            </div>
                            <div class="genre-card-name">${genre.label}</div>
                        </button>
                    `;
                }).join('');
                return `
                    <section class="genre-section">
                        ${section.groupName ? `<div class="genre-section-title">${section.groupName}</div>` : ''}
                        <div class="genre-card-grid">${cards}</div>
                    </section>
                `;
            }).join('');
        }

        window.openGenreDetail = async function(genreId) {
            const requestId = ++genreDetailRequestId;
            if (genreController) genreController.abort();
            genreController = new AbortController();
            const genre = findGenreById(genreId);
            genreDetailActiveId = genreId;
            genreDetailVisibleCount = 24;
            genreDetailMovies = [];
            let data;
            try {
                await loadGenreCatalog();
                data = await apiRequest(
                    `/api/genre/${encodeURIComponent(genreId)}?type=${encodeURIComponent(browseType)}`,
                    { signal: genreController.signal }
                );
            } catch (error) {
                if (isAbortError(error)) return;
                showToast(error.message || 'Could not load this genre');
                return;
            }
            if (requestId !== genreDetailRequestId) return;
            genreDetailMovies = data.movies || [];
            const count = data.genre?.count || 0;
            const hero = document.getElementById('genreDetailHero');
            const title = document.getElementById('genreDetailTitle');
            const meta = document.getElementById('genreDetailMeta');
            const summary = document.getElementById('genreDetailSummary');
            const grid = document.getElementById('genreDetailGrid');

            title.innerText = genre.label;
            meta.innerText = `${count} ${count === 1 ? 'title' : 'titles'} in the catalogue`;
            summary.innerHTML = `
                <div class="genre-detail-summary-inner">
                    <div class="genre-detail-kicker">${genre.group}</div>
                    <p>${genre.description}</p>
                    <div class="genre-detail-actions">
                    <button class="btn-request" type="button" onclick="showHome();">Browse all</button>
                    ${count === 0 ? `<button class="btn-request btn-sm-outline" type="button" onclick="requestMovie('${genre.label.replace(/'/g, "\\'")}')">Request title</button>` : ''}
                    </div>
                </div>
            `;

            hero.dataset.genre = genreId;
            hero.style.backgroundImage = `url("${GENRE_ARTWORK[genreId]}"), linear-gradient(135deg, ${getGenreVisualPalette(genre.group).from}, ${getGenreVisualPalette(genre.group).to})`;
            hero.style.backgroundSize = 'cover';
            hero.style.backgroundPosition = 'center';
            renderGenreDetailResults(genre, count);

            document.getElementById('exploreContent').style.display = 'none';
            document.getElementById('genreDetailPage').classList.add('open');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        };

        function renderGenreDetailResults(genre, count) {
            const grid = document.getElementById('genreDetailGrid');
            const visibleMovies = genreDetailMovies.slice(0, genreDetailVisibleCount);
            const hasMore = visibleMovies.length < genreDetailMovies.length;
            grid.innerHTML = `
                <div class="genre-catalogue-heading">
                    <div>
                        <span class="genre-catalogue-kicker">Catalogue</span>
                        <h3>${genre.label} titles</h3>
                    </div>
                    <span class="genre-catalogue-count">${count} ${count === 1 ? 'title' : 'titles'}</span>
                </div>
                ${visibleMovies.length
                    ? renderCards(visibleMovies, 'grid-card', false)
                    : '<div class="empty-search-state genre-empty" style="grid-column:1/-1;">Nothing here yet.<br><span>This canonical genre has no titles in the real catalogue.</span></div>'}
                ${hasMore ? `<button class="btn-request genre-load-more" type="button" onclick="loadMoreGenreResults()">Load more</button>` : ''}
            `;
        }

        window.loadMoreGenreResults = function() {
            if (!genreDetailMovies.length || genreDetailVisibleCount >= genreDetailMovies.length) return;
            genreDetailVisibleCount += 24;
            const genre = findGenreById(genreDetailActiveId);
            if (genre) renderGenreDetailResults(genre, genreDetailMovies.length);
        };

        window.closeGenreDetail = function() {
            document.getElementById('genreDetailPage').classList.remove('open');
            document.getElementById('exploreContent').style.display = 'block';
            genreDetailActiveId = null;
            genreDetailMovies = [];
            genreDetailVisibleCount = 24;
            renderExploreScreen();
            document.querySelectorAll('[data-browse-type]').forEach(button => button.classList.toggle('active', button.dataset.browseType === browseType));
            loadBrowseCollections();
        };

        function renderHeroProgress(total, activeIndex) {
            const dots = Array.from({ length: total }, (_, i) =>
                `<span class="hero-progress-dot ${i === activeIndex ? 'active' : ''}"></span>`
            ).join('');
            document.getElementById('heroSlider').insertAdjacentHTML('beforeend', `<div class="hero-progress">${dots}</div>`);
        }

        function showHeroSlide(index) {
            if (!heroItems.length) return;
            heroIndex = (index + heroItems.length) % heroItems.length;
            const movie = heroItems[heroIndex];
            const heroSlider = document.getElementById('heroSlider');
            const openHeroDetails = () => {
                if (!allMovies.some(item => String(item.id) === String(movie.id))) {
                    allMovies.push(movie);
                }
                openDetails(String(movie.id), false);
            };
            const heroListButton = heroSlider.querySelector('.round-button');
            if (heroListButton) {
                heroListButton.onclick = () => {
                    activeMovie = movie;
                    activeDetailsMovieId = String(movie.id);
                    toggleCurrentMyList();
                };
            }
            const heroArtwork = movie.backdrop || movie.image;
            heroSlider.style.backgroundImage = heroArtwork
                ? `url("${heroArtwork}"), ${IMAGE_FALLBACK_GRADIENT}`
                : IMAGE_FALLBACK_GRADIENT;
            document.getElementById('heroTitle').innerText = movie.title;
            document.getElementById('heroMeta').innerText = [movie.year, movie.category, movie.language].filter(Boolean).join(' • ');
            document.querySelector('#heroSlider .eyebrow').innerText = 'TRENDING NOW';
            document.getElementById('heroWatch').onclick = openHeroDetails;
            document.getElementById('heroInfo').onclick = openHeroDetails;
            heroSlider.querySelectorAll('.hero-progress-dot').forEach((dot, dotIndex) => dot.classList.toggle('active', dotIndex === heroIndex));
            const position = document.getElementById('heroPosition');
            if (position) position.textContent = `${heroIndex + 1} / ${heroItems.length}`;
        }

        function stepHero(direction) {
            if (!heroItems.length) return;
            showHeroSlide(heroIndex + direction);
            if (heroTimer) {
                clearInterval(heroTimer);
                heroTimer = setInterval(() => stepHero(1), 6500);
            }
        }
        window.stepHero = stepHero;

        function getRecentSearches() {
            try {
                const items = JSON.parse(localStorage.getItem('flimfybox_recent_searches') || '[]');
                return Array.isArray(items) ? items.slice(0, 6) : [];
            } catch (error) {
                return [];
            }
        }

        function saveRecentSearches(items) {
            localStorage.setItem('flimfybox_recent_searches', JSON.stringify(items.slice(0, 6)));
        }

        function addRecentSearch(term) {
            const value = term.trim();
            if (!value) return;
            const list = getRecentSearches().filter(item => item.toLowerCase() !== value.toLowerCase());
            list.unshift(value);
            saveRecentSearches(list);
        }

        function getRecentlyViewed() {
            try {
                const items = JSON.parse(localStorage.getItem('flimfybox_recently_viewed') || '[]');
                return Array.isArray(items) ? items : [];
            } catch (error) {
                return [];
            }
        }

        function saveRecentlyViewed(items) {
            localStorage.setItem('flimfybox_recently_viewed', JSON.stringify(items.slice(0, 8)));
        }

        function addRecentlyViewed(movie) {
            if (!movie || !movie.title) return;
            const items = getRecentlyViewed().filter(item => String(item.id) !== String(movie.id));
            items.unshift({
                id: movie.id,
                title: movie.title,
                image: movie.image,
                year: movie.year,
                category: movie.category,
                genre: movie.genre,
                source: movie.source || 'local'
            });
            saveRecentlyViewed(items);
        }

        function ensureHomeSectionRow(rowId, title, icon, caption) {
            let row = document.getElementById(rowId);
            if (row) return row;

            const mainContent = document.getElementById('mainContent');
            if (!mainContent) return null;

            const anchor = document.getElementById('rowHollywood') || document.getElementById('rowBollywood') || document.getElementById('rowAnime') || document.getElementById('rowActualNewReleases') || document.getElementById('rowTrending');
            const fallback = document.createElement('div');
            fallback.className = 'movie-row';
            fallback.id = rowId;
            fallback.innerHTML = `
                <div class="row-header">
                    <div class="row-header-left"><i class="fas fa-${icon}"></i> ${title}</div>
                    <span class="row-caption">${caption}</span>
                </div>
                <div class="horizontal-scroll" id="${rowId === 'rowUpcoming' ? 'upcomingScroll' : rowId + 'Scroll'}"></div>
            `;

            if (anchor) {
                mainContent.insertBefore(fallback, anchor);
            } else {
                mainContent.appendChild(fallback);
            }

            return fallback;
        }

        function renderHome(movies) {
            if (heroTimer) {
                clearInterval(heroTimer);
                heroTimer = null;
            }

            const recentViewed = getRecentlyViewed();
            const recentScroll = document.getElementById('recentScroll');
            const recentRow = document.getElementById('rowRecent');
            if (recentViewed.length) {
                recentScroll.innerHTML = renderCards(recentViewed, 'card', false);
                recentRow.style.display = '';
            } else {
                recentRow.style.display = 'none';
                recentScroll.innerHTML = '';
            }

            const upcomingRow = ensureHomeSectionRow('rowUpcoming', 'Upcoming', 'calendar-plus', 'Coming soon');
            const upcomingScroll = upcomingRow ? upcomingRow.querySelector('#upcomingScroll') : document.getElementById('upcomingScroll');
            if (upcomingRow && upcomingScroll) {
                upcomingRow.style.display = '';
                upcomingScroll.innerHTML = '<div class="loader">Loading upcoming…</div>';
                loadHomeSection('/api/upcoming?limit=18', 'upcoming')
                    .then(data => {
                        if (data.status !== 'success') throw new Error(data.message || 'Could not load upcoming');
                        const upcoming = Array.isArray(data.movies) ? data.movies : [];
                        upcoming.forEach(item => {
                            if (item && item.id) {
                                tmdbMoviesMap[item.id] = { ...item, source: 'tmdb', id: item.id };
                            }
                        });
                        upcomingScroll.innerHTML = upcoming.length
                            ? renderCards(upcoming, 'card', true)
                            : '<div class="search-empty-state">No upcoming titles right now.<br><span>Check back soon.</span></div>';
                        upcomingRow.style.display = upcoming.length ? '' : 'none';
                    })
                    .catch(error => {
                        console.error('Upcoming load failed:', error);
                        upcomingScroll.innerHTML = '<div class="search-empty-state">Upcoming is temporarily unavailable.</div>';
                        upcomingRow.style.display = '';
                    });
            }

            const catalogueRows = [
                {
                    row: 'rowHollywood',
                    target: 'hollywoodScroll',
                    matches: movie => {
                        const category = String(movie.category || '').toLowerCase();
                        const contentType = String(movie.content_type || '').toLowerCase();
                        const language = String(movie.language || '').toLowerCase();
                        const genre = String(movie.genre || '').toLowerCase();
                        if (/(anime|korean|japan|chinese)/.test(`${category} ${contentType} ${genre}`)) return false;
                        return /(^|[^a-z])hollywood([^a-z]|$)/.test(category)
                            || (
                                ['english', 'english movie', 'english movies', 'movie', 'movies', 'film', 'films'].includes(category)
                                && /(^|[^a-z])english([^a-z]|$)/.test(language)
                            );
                    }
                },
                {
                    row: 'rowBollywood',
                    target: 'bollywoodScroll',
                    matches: movie => {
                        const category = String(movie.category || '').toLowerCase();
                        const contentType = String(movie.content_type || '').toLowerCase();
                        const language = String(movie.language || '').toLowerCase();
                        const genre = String(movie.genre || '').toLowerCase();
                        if (/anime|korean|japan|chinese/.test(`${category} ${contentType} ${genre}`)) return false;
                        return /(^|[^a-z])(bollywood|hindi)([^a-z]|$)/.test(category)
                            || /(^|[^a-z])hindi([^a-z]|$)/.test(language);
                    }
                },
                {
                    row: 'rowAnime',
                    target: 'animeScroll',
                    matches: movie => /(^|[^a-z])anime([^a-z]|$)/.test(
                        `${movie.category || ''} ${movie.content_type || ''} ${movie.genre || ''}`.toLowerCase()
                    )
                }
            ];
            catalogueRows.forEach(({ row, target, matches }) => {
                const items = movies.filter(matches).slice(0, 12);
                const rowElement = document.getElementById(row);
                document.getElementById(target).innerHTML = items.length
                    ? renderCards(items, 'card', false)
                    : '';
                rowElement.style.display = items.length ? '' : 'none';
            });

            // These rows must not depend on the first paginated catalogue page.
            // Load their own bounded collections so older titles are visible
            // immediately without forcing users to scroll through the entire
            // catalogue first.
            loadHomeSection('/api/home/catalogue-rows', 'catalogue-rows')
                .then(data => {
                    if (data.status !== 'success') throw new Error(data.message || 'Could not load catalogue rows');
                    const collections = data.collections || {};
                    [
                        ['rowHollywood', 'hollywoodScroll', collections.hollywood],
                        ['rowBollywood', 'bollywoodScroll', collections.bollywood],
                        ['rowAnime', 'animeScroll', collections.anime]
                    ].forEach(([row, target, items]) => {
                        const list = Array.isArray(items) ? items : [];
                        document.getElementById(target).innerHTML = list.length
                            ? renderCards(list, 'card', false)
                            : '';
                        document.getElementById(row).style.display = list.length ? '' : 'none';
                    });
                })
                .catch(error => console.error('Catalogue rows load failed:', error));

            const requestId = ++newReleaseRequestId;
            loadHomeSection('/api/home/new-releases', 'new-releases')
                .then(data => {
                    if (requestId !== newReleaseRequestId) return;
                    if (data.status !== 'success') throw new Error(data.message || 'Could not load new releases');
                    const newReleases = data.movies || [];
                    const newReleasesRow = document.getElementById('rowActualNewReleases');
                    const newReleasesCaption = newReleasesRow.querySelector('.row-caption');
                    if (newReleasesCaption) {
                        const releaseWindow = Number(data.window_days) || 60;
                        newReleasesCaption.innerText = `Released in the last ${releaseWindow} days`;
                    }
                    document.getElementById('actualNewReleasesScroll').innerHTML = newReleases.length
                        ? renderCards(newReleases, 'card', false)
                        : '<div class="search-empty-state">No qualifying new releases right now.<br><span>Recently added catalogue titles are shown below.</span></div>';
                    newReleasesRow.style.display = '';
                })
                .catch(error => {
                    console.error('New release load failed:', error);
                    const fallbackRow = document.getElementById('rowActualNewReleases');
                    const fallbackCaption = fallbackRow?.querySelector('.row-caption');
                    if (fallbackCaption) fallbackCaption.innerText = 'Released in the last 60 days';
                    document.getElementById('actualNewReleasesScroll').innerHTML = '<div class="search-empty-state">No qualifying new releases right now.<br><span>Recently added catalogue titles are shown below.</span></div>';
                })
                .finally(completeInitialHomeLoadingStep);

            const trendingRequest = ++trendingRequestId;
            loadHomeSection('/api/home/trending?source=day', 'trending')
                .then(data => {
                    if (trendingRequest !== trendingRequestId) return;
                    if (data.status !== 'success') throw new Error(data.message || 'Could not load trending titles');
                    const trending = Array.isArray(data.results) ? data.results : [];
                    const trendingRow = document.getElementById('rowTrending');
                    document.getElementById('trendingScroll').innerHTML = trending.length
                        ? renderCards(trending.slice(0, 12), 'card', false)
                        : '';
                    trendingRow.style.display = trending.length ? '' : 'none';
                    const heroSlider = document.getElementById('heroSlider');
                    heroItems = trending.slice(0, Number(data.hero_limit || 10))
                        .filter((movie, index, items) => items.findIndex(item => String(item.id) === String(movie.id)) === index);
                    if (!heroItems.length) {
                        heroSlider.classList.add('is-loading');
                        document.querySelector('#heroSlider .eyebrow').innerText = 'TRENDING NOW';
                        document.getElementById('heroTitle').innerText = 'Trending titles aren\'t in the catalogue yet';
                        document.getElementById('heroMeta').innerText = 'TMDB ranking is available, but local catalogue coverage is empty';
                        return;
                    }
                    heroSlider.classList.remove('is-loading');
                    heroIndex = 0;
                    heroSlider.querySelector('.hero-progress')?.remove();
                    renderHeroProgress(heroItems.length, heroIndex);
                    showHeroSlide(heroIndex);
                    heroTimer = setInterval(() => stepHero(1), 6500);
                })
                .catch(error => {
                    console.error('Trending load failed:', error);
                    const heroSlider = document.getElementById('heroSlider');
                    heroSlider.classList.add('is-loading');
                    document.querySelector('#heroSlider .eyebrow').innerText = 'TRENDING NOW';
                    document.getElementById('heroTitle').innerText = 'Trending titles are temporarily unavailable';
                    document.getElementById('heroMeta').innerText = 'Please refresh in a moment';
                })
                .finally(completeInitialHomeLoadingStep);

            document.getElementById('moreGrid').innerHTML = renderCards(movies.slice(15), 'grid-card', false);
        }

        window.openCardDetails = function(movie, isTMDB) {
            if (!isTMDB && !allMovies.some(item => String(item.id) === String(movie.id))) {
                allMovies.push(movie);
            }
            openDetails(String(movie.id), isTMDB);
        };

        function renderCards(movies, cardClass = 'card', forceTMDB = false) {
            if (!movies.length) return '<div class="search-empty-state" style="grid-column:1/-1;">No titles to show right now.<br><span>Check back soon.</span></div>';
            return movies.map(m => {
                const isTMDB = forceTMDB || m.source === 'tmdb';
                const rating = m.rating && m.rating !== 'N/A' ? `⭐ ${m.rating}` : '';
                const badge = isTMDB
                    ? `<div class="card-badge request">${m.is_upcoming ? 'Upcoming' : 'Request'}</div>`
                    : (rating ? `<div class="card-badge available">${rating}</div>` : '');
                return `
                    <div class="${cardClass}" tabindex="0" role="button" aria-label="Open ${m.title}" onclick='openCardDetails(${JSON.stringify(m).replace(/'/g, "&#39;")}, ${isTMDB})' onkeydown="if(event.key==='Enter') openCardDetails(${JSON.stringify(m).replace(/'/g, "&#39;")}, ${isTMDB})">
                        <img src="${m.image || POSTER_PLACEHOLDER}" class="card-img" loading="lazy" onerror="this.onerror=null; this.src='${POSTER_PLACEHOLDER}'; this.classList.add('image-fallback')">
                        <div class="card-title">${m.title}</div>
                        <div class="card-meta"><span>${m.year || '—'}</span>${badge}</div>
                    </div>
                `;
            }).join('');
        }

// Hybrid Search
let searchTimeout;
let searchRequestId = 0;

document.getElementById('genreSearchInput')?.addEventListener('input', (e) => {
    renderExploreScreen();
});

window.triggerSearchAgain = function() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput && searchInput.value.trim() !== '') {
        searchInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
};

window.clearSearchInput = function() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.value = '';
        searchInput.dispatchEvent(new Event('input', { bubbles: true }));
        searchInput.focus();
    }
};

document.getElementById('searchInput').addEventListener('click', function(e) {
    if (this.value.trim() !== '') {
        this.dispatchEvent(new Event('input', { bubbles: true }));
    }
});

document.getElementById('searchInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        this.blur(); // Collapse keyboard on mobile after pressing Enter
    }
});

document.getElementById('searchInput').addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    if (searchController) searchController.abort();
    const q = e.target.value.trim();
    const dropdown = document.getElementById('searchDropdown');
    const clearBtn = document.getElementById('clearSearchBtn');

    if (clearBtn) {
        clearBtn.style.display = e.target.value.length > 0 ? 'block' : 'none';
    }

    if (!q) {
        renderSearchDiscovery();
        return;
    }

    dropdown.innerHTML = '<div class="loader">Searching catalog…</div>';
    dropdown.classList.add('active');

    searchTimeout = setTimeout(async () => {
        searchRequestId++;
        const currentId = searchRequestId;
        searchController = new AbortController();
        const signal = searchController.signal;
        try {
            const suggestionPromise = fetch(`/api/suggest?q=${encodeURIComponent(q)}`, { signal })
                .then(response => response.ok ? response.json() : [])
                .catch(error => {
                    if (isAbortError(error)) throw error;
                    console.warn('Search suggestions unavailable:', error);
                    return [];
                });
            const [suggestionResponse, response] = await Promise.all([
                suggestionPromise,
                fetch(`/api/search?q=${encodeURIComponent(q)}`, { signal })
            ]);
            const suggestions = Array.isArray(suggestionResponse) ? suggestionResponse : [];
            const searchData = await response.json();
            if (currentId !== searchRequestId) return;
            if (!response.ok || searchData.status !== 'success') throw new Error(searchData.message || 'Search failed');
            const results = searchData.results || [];
            results.forEach(r => {
                if (r.source === 'tmdb') tmdbMoviesMap[r.id] = r;
                else if (!allMovies.some(m => String(m.id) === String(r.id))) allMovies.push(r);
            });

            if (!results.length) {
                const suggestionMarkup = renderSuggestionMarkup(suggestions);
                addRecentSearch(q);
                dropdown.innerHTML = suggestionMarkup + `<div class="empty-search-state"><strong>No match found</strong><span>“${escapeHtml(q)}” isn't in the catalogue yet.</span><div style="margin-top:12px;"><button class="btn-sm btn-sm-primary" type="button" data-request-search><i class="fas fa-paper-plane"></i> Request this title</button></div></div>`;
                bindSearchSuggestions(dropdown, q);
                return;
            }

            addRecentSearch(q);
            dropdown.innerHTML = results.slice(0, 8).map(r => {
                const isTMDB = r.source === 'tmdb';
                const isUpcoming = Boolean(r.is_upcoming);
                const isAvailable = Boolean(r.is_available);
                const status = isUpcoming ? 'Upcoming' : (isAvailable ? 'Available' : 'Unavailable');
                const actionLabel = isUpcoming ? 'Notify' : (isAvailable ? 'View' : 'Request');
                return `<div class="search-item fade-in" data-result-id="${escapeHtml(r.id)}" data-result-tmdb="${isTMDB}">
                    <img src="${escapeHtml(r.image || POSTER_PLACEHOLDER)}" loading="lazy" onerror="this.onerror=null; this.src='${POSTER_PLACEHOLDER}'">
                    <div class="search-item-info"><div class="search-item-title">${escapeHtml(r.title)}</div>
                    <div class="search-item-meta"><span>${escapeHtml(r.year || '—')}</span><span class="status-pill ${isAvailable ? 'available' : 'request'}">${status}</span></div></div>
                    <div class="search-actions"><button class="btn-sm ${isAvailable ? 'btn-sm-primary' : 'btn-sm-outline'}" type="button">${actionLabel}</button></div></div>`;
            }).join('');
            bindSearchSuggestions(dropdown, q, results);
        } catch (error) {
            if (isAbortError(error)) return;
            console.error('Search failed:', error);
            if (currentId === searchRequestId) {
                dropdown.innerHTML = '<div class="search-empty-state"><strong>Search unavailable</strong><span>Please try again in a moment.</span></div>';
            }
        }
    }, 260);
});

function renderSuggestionMarkup(suggestions) {
    if (!Array.isArray(suggestions) || !suggestions.length) return '';
    return `<div class="search-suggestion-group"><div class="search-dropdown-label">Suggestions</div>${suggestions.slice(0, 6).map(item => `
        <button class="search-suggestion" type="button" data-suggestion="${escapeHtml(item)}">
            <i class="fas fa-magnifying-glass"></i><span>${escapeHtml(item)}</span>
        </button>`).join('')}</div>`;
}

function bindSearchSuggestions(dropdown, query, results = []) {
    dropdown.querySelectorAll('[data-suggestion]').forEach(button => {
        button.addEventListener('click', () => {
            const input = document.getElementById('searchInput');
            input.value = button.dataset.suggestion || '';
            input.dispatchEvent(new Event('input', { bubbles: true }));
        });
    });
    dropdown.querySelector('[data-request-search]')?.addEventListener('click', () => requestSilent(query));
    dropdown.querySelectorAll('[data-result-id]').forEach(item => {
        item.addEventListener('click', () => {
            const movie = results.find(result => String(result.id) === String(item.dataset.resultId));
            if (!movie) return;
            addRecentSearch(movie.title || query);
            openDetails(String(movie.id), item.dataset.resultTmdb === 'true');
            dropdown.classList.remove('active');
        });
    });
}


// Hide dropdown if clicked outside
document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('searchDropdown');
    const container = document.querySelector('.search-section');
    if (!container.contains(e.target)) {
        dropdown.classList.remove('active');
    }
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.getElementById('searchDropdown').classList.remove('active');
        closeTrailer();
        closeWebPlayer();
    }
});
        window.shareCurrentMovie = async function() {
            if (!activeMovie) {
                showToast('Open a movie first');
                return;
            }

            const title = String(activeMovie.title || 'this movie').trim();
            const movieId = activeMovie.id;
            const shareUrl = movieId
                ? `${window.location.origin}/webapp?movie=${encodeURIComponent(movieId)}`
                : window.location.href;
            const shareText = `Watch ${title} on FlimfyBox 🎬`;

            try {
                if (navigator.share) {
                    await navigator.share({ title: `${title} · FlimfyBox`, text: shareText, url: shareUrl });
                    return;
                }
            } catch (error) {
                if (error && error.name === 'AbortError') return;
            }

            const telegramShareUrl = `https://t.me/share/url?url=${encodeURIComponent(shareUrl)}&text=${encodeURIComponent(shareText)}`;
            try {
                if (tg && typeof tg.openTelegramLink === 'function') {
                    tg.openTelegramLink(telegramShareUrl);
                    return;
                }
            } catch (_error) {
                // Fall through to clipboard for browsers and older Telegram clients.
            }

            try {
                await navigator.clipboard.writeText(`${shareText}\n${shareUrl}`);
                showToast('Share link copied');
            } catch (_error) {
                showToast('Copy the movie link from your browser');
            }
        };

        // Details
        window.openDetails = function(id, isTMDB) {
            const movie = isTMDB ? tmdbMoviesMap[id] : allMovies.find(m => m.id == id);
            if (!movie) return;
            if (!isTMDB) trackRecommendationEvent('miniapp_open_details', movie.id);
            activeMovie = movie;
            activeDetailsMovieId = isTMDB ? String(id) : String(movie.id);
            addRecentlyViewed(movie);
            const requestId = ++detailsRequestId;
            if (detailsController) detailsController.abort();
            detailsController = new AbortController();
            const myListButton = document.getElementById('detailMyListButton');
            if (myListButton) {
                myListButton.innerHTML = savedMovieIds.has(String(movie.id))
                    ? '<i class="fas fa-check"></i>' : '<i class="fas fa-plus"></i>';
                refreshMyListButton(movie.id, myListButton);
            }
            const detailsPage = document.getElementById('detailsPage');
            const detailsBackdrop = document.getElementById('dpBackdrop');
            const detailsPoster = document.getElementById('dpFloatPoster');
            const detailsTitle = document.getElementById('dpTitle');
            const detailsRating = document.getElementById('dpRating');
            const detailsGenre = document.getElementById('dpGenre');
            const detailsDescription = document.getElementById('dpDesc');
            const detailsArtwork = movie.backdrop || movie.image;
            detailsBackdrop.style.backgroundImage = detailsArtwork
                ? `url("${detailsArtwork}"), ${IMAGE_FALLBACK_GRADIENT}`
                : IMAGE_FALLBACK_GRADIENT;
            detailsPoster.src = movie.image || POSTER_PLACEHOLDER;
            detailsPoster.onerror = () => {
                detailsPoster.onerror = null;
                detailsPoster.src = POSTER_PLACEHOLDER;
            };
            detailsTitle.innerText = movie.title || 'Loading details…';
            detailsRating.innerText = movie.rating && movie.rating !== 'N/A' ? movie.rating : '—';
            detailsGenre.innerText = movie.genre || movie.category || 'Catalogue title';
            detailsDescription.innerText = 'Loading story and availability…';
            document.getElementById('castSection').innerHTML = '';
            document.getElementById('dpTrailerBtn').innerHTML = '';
            if (movie.is_upcoming) {
                renderLockedCommunityRating(movie.title || 'this title');
            } else {
                renderCommunityRating(movie.id, movie.title || 'this title');
            }
            document.getElementById('dpSeasons').innerHTML = '<div class="dl-heading">Loading seasons…</div>';
            document.getElementById('dpLinks').innerHTML = '<div class="dl-heading">Loading available files…</div>';
            detailsPage.classList.add('open', 'is-loading');
            // The card already contains the title, poster, year, rating and genre.
            // Show that immediately while files/cast finish loading in the background.
            detailsDescription.innerText = movie.description || 'Details and availability are loading…';
            detailsPage.classList.remove('is-loading');
            if (isTMDB || movie.is_upcoming || movie.is_available === false) {
                const backdropImg = movie.backdrop || movie.image;
                document.getElementById('dpBackdrop').style.backgroundImage = backdropImg
                    ? `url("${backdropImg}"), ${IMAGE_FALLBACK_GRADIENT}`
                    : IMAGE_FALLBACK_GRADIENT;
                document.getElementById('dpFloatPoster').src = movie.image || POSTER_PLACEHOLDER;
                document.getElementById('dpTitle').innerText = movie.title;
                document.getElementById('dpRating').innerText = movie.rating && movie.rating !== 'N/A' ? movie.rating : '—';
                document.getElementById('dpGenre').innerText = movie.genre || 'Action, Drama';
                document.getElementById('dpDesc').innerText = movie.description || 'No description available.';
                document.getElementById('castSection').innerHTML = '';
                // Only show "upcoming" UI if the movie is genuinely pre-release.
                // A TMDB result that has already released should show the Request flow.
                const isGenuinelyUpcoming = movie.is_upcoming === true
                    || movie.availability_state === 'upcoming'
                    || (movie.release_date && movie.release_date > new Date().toISOString().slice(0, 10));
                if (isGenuinelyUpcoming) {
                    renderUpcomingActions(movie);
                } else {
                    renderUnavailableAction(movie);
                }
                detailsPage.classList.remove('is-loading');
                return;
            }

            const cachedDetails = movieDetailsCache.get(String(id));
            const detailsRequest = cachedDetails
                ? Promise.resolve({ status: 'success', movie: cachedDetails })
                : fetch(`/api/movie/${id}`, { signal: detailsController.signal })
                .then(async res => {
                    let data;
                    try {
                        data = await res.json();
                    } catch (_error) {
                        throw new Error('Details are temporarily unavailable.');
                    }
                    if (!res.ok || data.status !== 'success') {
                        throw new Error(data.message || 'Details are temporarily unavailable.');
                    }
                    return data;
                });

            detailsRequest
                .then(data => {
                    if (requestId !== detailsRequestId || String(activeMovie?.id) !== String(id)) return;
                    if (data.status === 'success') {
                        const m = data.movie;
                        movieDetailsCache.set(String(id), m);
                        activeMovie = { ...movie, ...m, source: 'local' };
                        addRecentlyViewed(activeMovie);
                        const backdropUrl = m.backdrop ? m.backdrop : m.image;
                        document.getElementById('dpBackdrop').style.backgroundImage = backdropUrl
                            ? `url("${backdropUrl}"), ${IMAGE_FALLBACK_GRADIENT}`
                            : IMAGE_FALLBACK_GRADIENT;
                        document.getElementById('dpFloatPoster').src = m.image || POSTER_PLACEHOLDER;
                        document.getElementById('dpTitle').innerText = m.title;
                        document.getElementById('dpRating').innerText = m.rating && m.rating !== 'N/A' ? m.rating : '—';
                        document.getElementById('dpGenre').innerText = m.genre || 'Drama';
                        document.getElementById('dpDesc').innerText = m.description || 'Story details are not available yet.';
                        if (m.cast && m.cast.trim().length > 0) {
                            const actors = m.cast.split(',');
                            let castHtml = '<div style="margin-bottom:20px;">';
                            actors.forEach(actor => {
                                const cleanName = actor.trim();
                                if (cleanName) castHtml += `<span class="cast-chip">${cleanName}</span>`;
                            });
                            castHtml += '</div>';
                            document.getElementById('castSection').innerHTML = castHtml;
                        } else {
                            document.getElementById('castSection').innerHTML = '';
                        }
                        document.getElementById('dpTrailerBtn').innerHTML = '';
                        if (m.is_upcoming) {
                            renderLockedCommunityRating(m.title || movie.title || 'this title');
                        } else {
                            renderCommunityRating(m.id, m.title || movie.title || 'this title');
                        }

                        const seasonsContainer = document.getElementById('dpSeasons');
                        const linksContainer = document.getElementById('dpLinks');
                        seasonsContainer.innerHTML = '';
                        linksContainer.innerHTML = '';

                        if (m.is_upcoming) {
                            renderUpcomingActions(m);
                        } else if (m.files && m.files.length) {
                            let hasSeasons = false;
                            const seasonsMap = {};
                            const movieFiles = [];

                            m.files.forEach(f => {
                                const info = `${f.extra_info || ''} ${f.quality || ''}`;
                                const { season: s, episodeStart: e, episodeEnd } =
                                    parseMiniAppEpisodeMetadata(info);
                                if (s !== null) {
                                    hasSeasons = true;
                                    if (!seasonsMap[s]) seasonsMap[s] = { episodes: {} };
                                    const sortEp = e !== null ? e : 0;
                                    const isCompleteSeason = e === null;
                                    const rangeLabel = episodeEnd !== null
                                        ? `${String(e).padStart(2, '0')}-${String(episodeEnd).padStart(2, '0')}`
                                        : '';
                                    const episodeLabel = isCompleteSeason
                                        ? 'Complete Season'
                                        : (episodeEnd !== null
                                            ? `Episode ${rangeLabel} Combined`
                                            : `Episode ${String(e).padStart(2, '0')}`);
                                    if (!seasonsMap[s].episodes[sortEp]) {
                                        seasonsMap[s].episodes[sortEp] = {
                                            title: episodeLabel,
                                            episodeStart: e,
                                            episodeEnd,
                                            isCompleteSeason,
                                            qualities: []
                                        };
                                    }
                                    seasonsMap[s].episodes[sortEp].qualities.push(f);
                                } else {
                                    movieFiles.push(f);
                                }
                            });

                            if (hasSeasons) {
                                const seasonNumbers = Object.keys(seasonsMap).map(Number).sort((a, b) => a - b);
                                let seasonsHtml = `<div class="season-scroll-wrapper"><div class="season-pill-container" id="seasonPillContainer">`;
                                seasonNumbers.forEach(sn => {
                                    seasonsHtml += `<div class="season-pill" data-season="${sn}" onclick="selectSeason(${m.id}, ${sn})">Season ${sn}</div>`;
                                });
                                seasonsHtml += `</div></div>`;
                                seasonsContainer.innerHTML = seasonsHtml;
                                window.currentMovieSeasons = seasonsMap;
                                selectSeason(m.id, seasonNumbers[0]);
                            } else {
                                let links = '<div class="dl-heading">Available qualities</div>';
                                m.files.forEach(f => {
                                    links += `
                                        <button class="dl-btn" onclick="downloadMovie(${m.id}, ${f.id})">
                                            <span class="quality-text"><i class="fas fa-download"></i> ${f.quality} <span class="file-size">${f.size || 'N/A'}</span></span>
                                            <span class="action">Download</span>
                                        </button>
                                    `;
                                });
                                linksContainer.innerHTML = links;
                            }
                        } else {
                            linksContainer.innerHTML = `
                                <div class="dl-heading">Download</div>
                                <button class="dl-btn" onclick="downloadMovie(${m.id})">
                                    <span class="quality-text"><i class="fas fa-download"></i> 1080p Full HD</span>
                                    <span class="action">Download</span>
                                </button>
                            `;
                        }
                        detailsPage.classList.remove('is-loading');
                    }
                })
                .catch(error => {
                    if (requestId !== detailsRequestId || String(activeMovie?.id) !== String(id)) return;
                    if (isAbortError(error)) return;
                    console.error('Details load failed:', error);
                    detailsDescription.innerText = 'Details are temporarily unavailable. You can go back and try again.';
                    document.getElementById('dpLinks').innerHTML = '<button class="btn-request" type="button" onclick="retryActiveDetails()"><i class="fas fa-rotate-right"></i> Retry details</button>';
                    detailsPage.classList.remove('is-loading');
                });
        };
        window.retryActiveDetails = function() {
            if (!activeMovie) return;
            openDetails(String(activeMovie.id), false);
        };
        function renderUpcomingActions(movie) {
            const actionContainer = document.getElementById('dpTrailerBtn');
            const linksContainer = document.getElementById('dpLinks');
            if (!actionContainer) return;
            const tmdbId = String(movie.tmdb_id || movie.id || '').replace(/^tmdb_/, '');
            const releaseDate = movie.release_date || '';
            const renderNotificationButton = (stage, enabled) => {
                const availableStage = stage === 'availability';
                const label = availableStage
                    ? (enabled ? 'Download notification set' : 'Notify when available for download')
                    : (enabled ? 'Notification Set' : 'Notify when released');
                const icon = enabled ? 'fa-check' : 'fa-bell';
                return `<button class="btn-request upcoming-notify-button" type="button" data-stage="${stage}" data-tmdb-id="${escapeHtml(tmdbId)}" data-release-date="${escapeHtml(releaseDate)}" data-title="${escapeHtml(movie.title || '')}" onclick="toggleUpcomingNotification(this)">
                    <i class="fas ${icon}"></i> <span>${label}</span>
                </button>`;
            };
            actionContainer.innerHTML = renderNotificationButton('release', false);
            if (linksContainer) {
                linksContainer.innerHTML = releaseDate
                    ? `<div class="pre-release-note"><i class="fas fa-calendar"></i> Expected release: ${escapeHtml(releaseDate)}</div>`
                    : '<div class="pre-release-note"><i class="fas fa-calendar"></i> Release date is not available yet.</div>';
            }
            const button = actionContainer.querySelector('.upcoming-notify-button');
            if (!button || !tmdbId || !releaseDate) return;
            fetch(`/api/upcoming/reminder?tmdb_id=${encodeURIComponent(tmdbId)}&release_date=${encodeURIComponent(releaseDate)}&title=${encodeURIComponent(movie.title || '')}`, {
                headers: telegramAuthHeaders()
            })
                .then(response => response.json().then(data => ({ ok: response.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok || data.status !== 'success') throw new Error(data.message || 'Notification status unavailable');
                    const stage = data.availability_state === 'upcoming' ? 'release' : 'availability';
                    button.dataset.stage = stage;
                    setUpcomingNotificationButton(
                        button,
                        stage === 'release'
                            ? Boolean(data.release_notification_set)
                            : Boolean(data.availability_notification_set),
                    );
                })
                .catch(error => console.warn('Upcoming notification status unavailable:', error));
        }
        function renderUnavailableAction(movie) {
            const actionContainer = document.getElementById('dpTrailerBtn');
            const linksContainer = document.getElementById('dpLinks');
            
            const tmdbId = String(movie.tmdb_id || movie.id || '').replace(/^tmdb_/, '');
            const releaseDate = movie.release_date || '';
            
            const renderNotificationButton = (stage, enabled) => {
                const availableStage = stage === 'availability';
                const label = availableStage
                    ? (enabled ? 'Download notification set' : 'Notify when available for download')
                    : (enabled ? 'Notification Set' : 'Notify when released');
                const icon = enabled ? 'fa-check' : 'fa-bell';
                return `<button class="btn-request upcoming-notify-button" style="margin-top: 10px;" type="button" data-stage="${stage}" data-tmdb-id="${escapeHtml(tmdbId)}" data-release-date="${escapeHtml(releaseDate)}" data-title="${escapeHtml(movie.title || '')}" onclick="toggleUpcomingNotification(this)">
                    <i class="fas ${icon}"></i> <span>${label}</span>
                </button>`;
            };

            if (actionContainer) {
                const requestBtnHtml = `<button class="btn-request" type="button" onclick="requestSilent('${String(movie.title || '').replace(/'/g, "\\'")}')">
                    <i class="fas fa-paper-plane"></i> Request ${escapeHtml(movie.title || 'this title')}
                </button>`;
                
                actionContainer.innerHTML = requestBtnHtml + renderNotificationButton('availability', false);
                
                const button = actionContainer.querySelector('.upcoming-notify-button');
                if (button && tmdbId) {
                    fetch(`/api/upcoming/reminder?tmdb_id=${encodeURIComponent(tmdbId)}&release_date=${encodeURIComponent(releaseDate)}&title=${encodeURIComponent(movie.title || '')}`, {
                        headers: telegramAuthHeaders()
                    })
                        .then(response => response.json().then(data => ({ ok: response.ok, data })))
                        .then(({ ok, data }) => {
                            if (!ok || data.status !== 'success') throw new Error(data.message || 'Notification status unavailable');
                            setUpcomingNotificationButton(button, Boolean(data.availability_notification_set));
                        })
                        .catch(error => console.warn('Upcoming notification status unavailable:', error));
                }
            }
            if (linksContainer) {
                linksContainer.innerHTML = '<div class="pre-release-note"><i class="fas fa-circle-info"></i> This title is currently unavailable on FlimfyBox.</div>';
            }
        }
        function setUpcomingNotificationButton(button, enabled) {
            if (!button) return;
            button.classList.toggle('is-enabled', enabled);
            const availableStage = button.dataset.stage === 'availability';
            const label = availableStage
                ? (enabled ? 'Download notification set' : 'Notify when available for download')
                : (enabled ? 'Notification Set' : 'Notify when released');
            button.innerHTML = enabled
                ? `<i class="fas fa-check"></i> <span>${label}</span>`
                : `<i class="fas fa-bell"></i> <span>${label}</span>`;
        }
        window.toggleUpcomingNotification = function(button) {
            if (!button || button.disabled) return;
            button.disabled = true;
            const body = {
                tmdb_id: button.dataset.tmdbId,
                stage: button.dataset.stage || 'release',
                title: button.dataset.title,
                release_date: button.dataset.releaseDate
            };
            fetch('/api/upcoming/reminder', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...telegramAuthHeaders() },
                body: JSON.stringify(body)
            })
                .then(response => response.json().then(data => ({ ok: response.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok || data.status !== 'success') throw new Error(data.message || 'Could not enable notification');
                    setUpcomingNotificationButton(button, true);
                    showToast(data.already_enabled ? '✓ Notification already set' : '✓ Notification set');
                })
                .catch(error => showToast(error.message))
                .finally(() => { button.disabled = false; });
        };
        function renderLockedCommunityRating(title) {
            const container = document.getElementById('communityRating');
            if (!container) return;
            container.dataset.selectedRating = '';
            container.dataset.submitting = 'false';
            container.innerHTML = `
                <h2 class="community-rating-title">Rate ${escapeHtml(title)}</h2>
                <div class="community-rating-stars is-locked" aria-label="Rating locked">
                    <span class="rating-lock"><i class="fas fa-lock"></i></span>
                    <span class="community-rating-locked-copy">Rating available after release</span>
                </div>
                <div class="community-rating-summary"><strong>Community Rating</strong><br>Rating available after release.</div>
            `;
        }
        window.renderCommunityRating = function(movieId, title) {
                const container = document.getElementById('communityRating');
                if (!container) return;
                container.innerHTML = '<h2 class="community-rating-title"></h2><div class="community-rating-stars"></div><button type="button" class="community-rating-submit" disabled>Submit rating</button><div class="community-rating-summary">Loading rating…</div>';
                container.dataset.selectedRating = '';
                container.querySelector('.community-rating-title').textContent = `Rate ${title}`;
                const stars = container.querySelector('.community-rating-stars');
                for (let value = 1; value <= 5; value += 1) {
                    const button = document.createElement('button');
                    button.type = 'button';
                    button.className = 'community-rating-star';
                    button.dataset.rating = String(value);
                    button.setAttribute('aria-label', `Rate ${value} out of 5`);
                    button.textContent = '☆';
                    button.addEventListener('mouseenter', () => previewCommunityStars(container, value));
                    button.addEventListener('mouseleave', () => previewCommunityStars(container, Number(container.dataset.selectedRating || 0)));
                    button.addEventListener('click', () => selectCommunityRating(value, container));
                    stars.appendChild(button);
                }
                container.querySelector('.community-rating-submit').addEventListener(
                    'click',
                    () => submitCommunityRating(movieId, container)
                );
                fetch(`/api/movie/${encodeURIComponent(movieId)}/rating`, {
                    headers: telegramAuthHeaders()
                })
                    .then(response => response.json().then(data => ({ ok: response.ok, data })))
                    .then(({ ok, data }) => {
                        if (!ok || data.status !== 'success') throw new Error(data.message || 'Rating unavailable');
                        updateCommunityRating(container, data);
                    })
                    .catch(() => {
                        container.querySelector('.community-rating-summary').textContent = 'Rating temporarily unavailable';
                    });
        };
        function previewCommunityStars(container, value) {
                container.querySelectorAll('.community-rating-star').forEach((button) => {
                    const selected = Number(button.dataset.rating) <= value;
                    button.classList.toggle('preview', selected);
                    button.textContent = selected ? '★' : '☆';
                });
        }
        function selectCommunityRating(value, container) {
                container.dataset.selectedRating = String(value);
                previewCommunityStars(container, value);
                const submit = container.querySelector('.community-rating-submit');
                if (submit) submit.disabled = false;
                const summary = container.querySelector('.community-rating-summary');
                if (summary) summary.textContent = `You selected ${value} / 5. Tap Submit rating to save it.`;
        }
        function updateCommunityRating(container, data) {
                const own = data.user_rating;
                container.querySelectorAll('.community-rating-star').forEach((button) => {
                    const selected = own && Number(button.dataset.rating) <= own;
                    button.classList.toggle('selected', Boolean(selected));
                    button.textContent = selected ? '★' : '☆';
                    button.disabled = !data.can_rate;
                });
                container.dataset.selectedRating = data.user_rating ? String(data.user_rating) : '';
                const submit = container.querySelector('.community-rating-submit');
                if (submit) submit.disabled = !data.can_rate || !data.user_rating;
                const summary = container.querySelector('.community-rating-summary');
                summary.innerHTML = data.count
                    ? `<strong>FlimfyBox Rating</strong><br>${Number(data.average).toFixed(1)} / 5 · ${data.count} rating${data.count === 1 ? '' : 's'}${own ? `<br>Your rating: ${own} / 5` : ''}`
                    : '<strong>FlimfyBox Rating</strong><br>No ratings yet';
        }
        function submitCommunityRating(movieId, container) {
                if (container.dataset.submitting === 'true') return;
                const value = Number(container.dataset.selectedRating || 0);
                if (!value) {
                    container.querySelector('.community-rating-summary').textContent = 'Select a star first.';
                    return;
                }
                container.dataset.submitting = 'true';
                container.querySelectorAll('.community-rating-star').forEach(button => { button.disabled = true; });
                const submit = container.querySelector('.community-rating-submit');
                if (submit) submit.disabled = true;
                container.querySelector('.community-rating-summary').textContent = 'Saving your rating…';
                fetch(`/api/movie/${encodeURIComponent(movieId)}/rating`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        ...telegramAuthHeaders()
                    },
                    body: JSON.stringify({ rating: value })
                }).then(response => response.json().then(data => ({ ok: response.ok, data })))
                    .then(({ ok, data }) => {
                        if (!ok || data.status !== 'success') throw new Error(data.message || 'Could not save rating');
                        trackRecommendationEvent('rating_submitted', movieId, { rating: value });
                        updateCommunityRating(container, data);
                    })
                    .catch(error => {
                        container.querySelector('.community-rating-summary').textContent = error.message;
                        container.querySelectorAll('.community-rating-star').forEach(button => { button.disabled = false; });
                        const submit = container.querySelector('.community-rating-submit');
                        if (submit) submit.disabled = !container.dataset.selectedRating;
                    })
                    .finally(() => { container.dataset.submitting = 'false'; });
        }
        function normalizeTrailerKey(value) {
            const raw = String(value || '').trim();
            if (!raw) return '';
            const match = raw.match(
                /(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)([A-Za-z0-9_-]{6,})/
            );
            return match ? match[1] : raw;
        }

        window.playCurrentTrailer = async function() {
            if (!activeMovie) return;
            const trailerKey = normalizeTrailerKey(activeMovie.trailer_key);
            if (trailerKey) {
                playTrailer(trailerKey);
                return;
            }
            const movieId = activeMovie.id;
            if (!movieId || String(movieId).startsWith('tmdb_')) {
                showToast('Trailer is not available for this title');
                return;
            }
            const button = document.querySelector('.detail-play');
            if (button) {
                button.disabled = true;
                button.classList.add('is-loading');
            }
            try {
                const response = await fetch(`/api/movie/${encodeURIComponent(movieId)}`);
                const data = await response.json();
                if (String(activeMovie?.id) !== String(movieId)) return;
                const details = data && data.status === 'success' ? data.movie : null;
                activeMovie = details ? { ...activeMovie, ...details } : activeMovie;
                const fetchedKey = normalizeTrailerKey(details?.trailer_key);
                if (fetchedKey) {
                    playTrailer(fetchedKey);
                    return;
                }
            } catch (error) {
                console.error('Trailer lookup failed:', error);
            } finally {
                if (button) {
                    button.disabled = false;
                    button.classList.remove('is-loading');
                }
            }
            showToast('Trailer is not available for this title');
        };


        window.selectSeason = function(movieId, seasonNum) {
            // Update Active Pill
            document.querySelectorAll('.season-pill').forEach(el => {
                if (parseInt(el.getAttribute('data-season')) === seasonNum) {
                    el.classList.add('active');
                    // Scroll into view
                    el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                } else {
                    el.classList.remove('active');
                }
            });

            const seasonData = window.currentMovieSeasons[seasonNum];
            const linksContainer = document.getElementById('dpLinks');
            
            if (!seasonData || Object.keys(seasonData.episodes).length === 0) {
                linksContainer.innerHTML = `<div class="empty-season">No episodes available for this season yet.</div>`;
                return;
            }

            const epNumbers = Object.keys(seasonData.episodes).map(Number).sort((a, b) => a - b);
            const episodeCount = Object.values(seasonData.episodes).reduce((count, ep) => {
                if (ep.isCompleteSeason) return count;
                if (ep.episodeEnd !== null && ep.episodeEnd !== undefined) {
                    return count + ep.episodeEnd - ep.episodeStart + 1;
                }
                return count + 1;
            }, 0);
            const seasonSummary = episodeCount
                ? `${episodeCount} EPISODES`
                : 'COMPLETE SEASON';
            let html = `<div class="dl-heading">SEASON ${seasonNum} • ${seasonSummary}</div><div class="episodes-list">`;
            
            epNumbers.forEach(epNum => {
                const ep = seasonData.episodes[epNum];
                const isCompleteSeason = Boolean(ep.isCompleteSeason);
                const isEpisodeRange = ep.episodeEnd !== null && ep.episodeEnd !== undefined;
                const epDisplayNum = isCompleteSeason
                    ? '—'
                    : (isEpisodeRange
                        ? `${String(ep.episodeStart).padStart(2, '0')}-${String(ep.episodeEnd).padStart(2, '0')}`
                        : String(ep.episodeStart ?? epNum).padStart(2, '0'));
                html += `
                <div class="episode-card">
                    <div class="ep-header">
                        ${isCompleteSeason
                            ? '<div class="ep-title">Complete Season</div>'
                            : (isEpisodeRange
                                ? `<div class="ep-title">Episode ${epDisplayNum} Combined</div>`
                                : `<div class="ep-number-group">
                                    <div class="ep-number-label">EPISODE</div>
                                    <div class="ep-number">${epDisplayNum}</div>
                                </div>`)}
                    </div>
                    <div class="ep-qualities">`;
                
                ep.qualities.forEach(q => {
                    const qualityLabel = escapeHtml(formatMiniAppQuality(q.quality) || 'Download');
                    const fileSize = escapeHtml(q.size || '');
                    html += `
                        <button class="ep-dl-btn" onclick="downloadMovie(${movieId}, ${q.id})">
                            <span class="ep-qtext"><i class="fas fa-play-circle"></i> ${qualityLabel} <span class="ep-size">${fileSize}</span></span>
                            <span class="ep-action"><i class="fas fa-download"></i></span>
                        </button>
                    `;
                });
                
                html += `</div></div>`;
            });
            html += `</div>`;
            
            linksContainer.innerHTML = html;
        };

        window.closeDetails = function() {
            document.getElementById('detailsPage').classList.remove('open');
        };

        window.playTrailer = function(key) {
            document.getElementById('trailerIframe').src = `https://www.youtube.com/embed/${key}?autoplay=1&rel=0`;
            document.getElementById('trailerModal').classList.add('active');
        };

        window.closeTrailer = function() {
            document.getElementById('trailerIframe').src = '';
            document.getElementById('trailerModal').classList.remove('active');
        };

        // In-App Web Player Modal Logic
        window.openWebPlayer = function(tmdbId) {
            document.getElementById('searchDropdown').classList.remove('active');
            const modal = document.getElementById('webPlayerModal');
            const iframeCont = document.getElementById('wpIframeContainer');
            const titleEl = document.getElementById('wpTitle');
            
            modal.classList.add('active');
            titleEl.innerText = 'Loading Player...';
            iframeCont.innerHTML = '<div class="wp-loader"><div class="loader"></div>Fetching Secure Stream...</div>';
            
            // Background async call to fetch IMDb ID
            fetch(`/api/imdb_id/${tmdbId}`)
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success' && data.imdb_id) {
                        titleEl.innerText = 'Secure Player · Premium Stream';
                        // streamimdb.ru requires IMDb ID for playing
                        iframeCont.innerHTML = `<iframe src="https://streamimdb.ru/embed/movie/${data.imdb_id}" allowfullscreen allow="autoplay"></iframe>`;
                    } else {
                        titleEl.innerText = 'Error loading stream';
                        iframeCont.innerHTML = '<div style="color:white;text-align:center;">❌ Could not find streaming source. Try Requesting the movie instead.</div>';
                    }
                })
                .catch(e => {
                    titleEl.innerText = 'Network Error';
                    iframeCont.innerHTML = '<div style="color:white;text-align:center;">❌ Network error while loading player.</div>';
                });
        };

        window.closeWebPlayer = function() {
            document.getElementById('webPlayerModal').classList.remove('active');
            document.getElementById('wpIframeContainer').innerHTML = ''; // Stop video playback
        };

        window.requestMovie = function(title) {
            tg.HapticFeedback.notificationOccurred('success');
            showToast('⏳ Requesting...');
            const user = tg.initDataUnsafe?.user || {id: 0, username: 'webapp', first_name: 'User'};
            fetch('/api/request', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({title, user_id: user.id, username: user.username, first_name: user.first_name})
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success') showToast('✅ Request sent!');
                else showToast('❌ Failed');
            })
            .catch(() => showToast('❌ Error'));
        };

        // 🔥 NAYA: Silent Request (Jab TMDB aur Google dono fail ho jayein)
        window.requestSilent = function(title) {
            tg.HapticFeedback.notificationOccurred('success');
            showToast('⏳ Sending Request...');
            const user = tg.initDataUnsafe?.user || {id: 0, username: 'webapp', first_name: 'User'};
            
            fetch('/api/request', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({title: title, user_id: user.id, username: user.username, first_name: user.first_name})
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success') {
                    showToast('✅ Request Sent to Admin!');
                    // Request bhejte hi Mini app close kar do (Seamless feel ke liye)
                    setTimeout(() => { tg.close(); }, 1500);
                } else {
                    showToast('❌ Failed to send');
                }
            })
            .catch(() => showToast('❌ Network Error'));
        };

        // 🛡️ NAYA: Anti-Bot Middleware Par Bhejne Wala Function
        window.downloadBot = function(id) {
            tg.HapticFeedback.impactOccurred('heavy');
            // Seedha Bot ki jagah pehle Secure verification page par bhejenge
            tg.openLink(`${window.location.origin}/watch/${id}`);
        };

        window.downloadMovie = function(id, fileId = null) {
            tg.HapticFeedback.impactOccurred('heavy');
            trackRecommendationEvent('miniapp_download', id, { file_id: fileId });
            const filePath = fileId ? `/file/${fileId}` : '';
            tg.openLink(`${window.location.origin}/watch/${id}${filePath}`);
        };

        // Start
        syncTelegramBackButton();
        setupMotionEffects();
        startInitialHomeLoading();
        loadMovies();
        
        // 🪄 NAYA JUGAD: URL se query nikal kar auto-search karna
        setTimeout(() => {
            const urlParams = new URLSearchParams(window.location.search);
            const reqQuery = urlParams.get('req');
            const sharedMovieId = urlParams.get('movie');
            
            if (reqQuery) {
                const requestedTitle = reqQuery.trim().slice(0, 200);
                const searchInput = document.getElementById('searchInput');
                searchInput.value = requestedTitle;
                showToast("🔍 Finding correct spelling...");
                // Search ko trigger karo
                searchInput.dispatchEvent(new Event('input', { bubbles: true }));
                const confirmRequest = () => window.requestMovie(requestedTitle);
                if (typeof tg.showPopup === 'function') {
                    tg.showPopup({
                        title: 'Request this title?',
                        message: `Send a request for "${requestedTitle}"?`,
                        buttons: [
                            { id: 'request', type: 'ok', text: 'Request' },
                            { id: 'cancel', type: 'cancel' }
                        ]
                    }, button => {
                        if (button && button.id === 'request') confirmRequest();
                    });
                } else if (window.confirm(`Send a request for "${requestedTitle}"?`)) {
                    confirmRequest();
                }
            }
            if (sharedMovieId) {
                const cachedMovie = allMovies.find(movie => String(movie.id) === String(sharedMovieId));
                if (cachedMovie) {
                    openDetails(String(sharedMovieId), false);
                } else {
                    fetch(`/api/movie/${encodeURIComponent(sharedMovieId)}`)
                        .then(response => response.ok ? response.json() : null)
                        .then(data => {
                            if (!data || data.status !== 'success' || !data.movie) return;
                            const sharedMovie = { ...data.movie, source: 'local' };
                            allMovies = [sharedMovie, ...allMovies.filter(movie => String(movie.id) !== String(sharedMovieId))];
                            openDetails(String(sharedMovieId), false);
                        })
                        .catch(() => showToast('This movie is temporarily unavailable'));
                }
            }
        }, 500); // Thoda ruk kar karenge taaki app load ho jaye
