from django.urls import path, re_path
from games import views


urlpatterns = [
    re_path(r'^game-categories/$', 
        views.GameCategoryList.as_view(), 
        name=views.GameCategoryList.name),
    re_path(r'^game-categories/(?P<pk>[0-9]+)/$', 
        views.GameCategoryDetail.as_view(),
        name=views.GameCategoryDetail.name),
    re_path(r'^games/$', 
        views.GameList.as_view(),
        name=views.GameList.name),
    re_path(r'^games/(?P<pk>[0-9]+)/$', 
        views.GameDetail.as_view(),
        name=views.GameDetail.name),
    re_path(r'^players/$', 
        views.PlayerList.as_view(),
        name=views.PlayerList.name),
    re_path(r'^players/(?P<pk>[0-9]+)/$', 
        views.PlayerDetail.as_view(),
        name=views.PlayerDetail.name),
    re_path(r'^player-scores/$', 
        views.PlayerScoreList.as_view(),
        name=views.PlayerScoreList.name),
    re_path(r'^player-scores/(?P<pk>[0-9]+)/$', 
        views.PlayerScoreDetail.as_view(),
        name=views.PlayerScoreDetail.name),
    re_path(r'^users/$',
        views.UserList.as_view(),
        name=views.UserList.name),
    re_path(r'^users/(?P<pk>[0-9]+)/$',
        views.UserDetail.as_view(),
        name=views.UserDetail.name),
    re_path(r'^$',
        views.ApiRoot.as_view(),
        name=views.ApiRoot.name),
]