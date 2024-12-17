import uvicorn
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app=FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates=Jinja2Templates(directory='templates')
df_anime=pd.read_csv('./static/data/filtered_dataset.csv')

#homepage
@app.get('/',response_class=HTMLResponse)
async def home(request: Request):
    featured = [38000, 1735, 40748, 21, 1535]
    action = [269, 30276, ]
    adventure = [11061, ]
    isekai = [76, 84, 95, 114, 45]
    
    featured_animes = df_anime[df_anime['anime_id'].isin(featured)].to_dict(orient='records')
    action_animes = df_anime[df_anime['anime_id'].isin(action)].to_dict(orient='records')
    adventure_animes = df_anime[df_anime['anime_id'].isin(adventure)].to_dict(orient='records')
    isekai_animes = df_anime[df_anime['anime_id'].isin(isekai)].to_dict(orient='records')
    return templates.TemplateResponse(
        'index.html',
        {"request": request,
         "featured": featured_animes,
         "action": action_animes,
         "adventure": adventure_animes,
         "isekai": isekai_animes}
    )

#load anime
@app.get('/anime/{anime_id}',response_class=HTMLResponse)
async def anime(request: Request, anime_id: int):
    anime=df_anime[df_anime['anime_id'] == anime_id].iloc[0]
    return templates.TemplateResponse(
        'animePage.html',
        {"request": request,
         "anime": anime}
    )

#by genre classification
@app.get('/animeList/{genre}',response_class=HTMLResponse)
async def anime_list(request: Request, genre: str):
    genre_df=df_anime[df_anime['Genres'].apply(lambda x: genre in x)]
    genre_df=genre_df[:20]
    genre_df=genre_df.sort_values(by='Rating',ascending=False)
    filtered_anime=genre_df.to_dict(orient="records")
    return templates.TemplateResponse(
        'animeList.html',
        {"request": request,
         "results": filtered_anime}
    )

# Search route
@app.get('/search', response_class=HTMLResponse)
async def search(request: Request, query:str=''):
    # query = request.args.get('q', '')
    if query:
        filtered_df = df_anime[df_anime['Name'].str.contains(query, case=False, na=False)]
        search_results = filtered_df.to_dict(orient='records')
    else:
        search_results = []
    return templates.TemplateResponse(
        'search_results.html',
        {"request": request,
         "results": search_results}
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)