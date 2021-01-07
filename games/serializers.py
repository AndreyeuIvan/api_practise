from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Game, GameCategory, Player, PlayerScore
import games.views


'''The GameSerializer class declares the attributes that represent the fields that we want to
be serialized. Notice that they have omitted the created attribute that was present in the
Game model. When there is a call to the inherited save method for this class, the overridden
create and update methods define how to create or modify an instance. In fact, these
methods must be implemented in our class because they just raise a
NotImplementedError exception in their base declaration.
The create method receives the validated data in the validated_data argument. The
code creates and returns a new Game instance based on the received validated data.
The update method receives an existing Game instance that is being updated and the new
validated data in the instance and validated_data arguments. The code updates the
values for the attributes of the instance with the updated attribute values retrieved from the
validated data, calls the save method for the updated Game instance and returns the
updated and saved instance.
В первой главе мы наследовали Сериалайзер
Во второй наследуем модель сериалайзер 
There is no need to override either create or update methods because the generic
behavior will be enough in this case. The ModelSerializer superclass provides
implementations for both methods.

The HyperlinkedModelSerializer is a type of ModelSerializer that uses hyperlinked
relationships instead of primary key relationships, and therefore, it represents the
realationships to other model instances with hyperlinks instead of primary key values. In
addition, the HyperlinkedModelSerializer generated a field named url with the URL
for the resource as its value. As seen in the case of ModelSerializer , the
HyperlinkedModelSerializer class provides default implementations for the create
and update methods.'''

class GameCategorySerializer(serializers.HyperlinkedModelSerializer):
    games = serializers.HyperlinkedRelatedField(
        many=True,
        read_only=True,
        view_name='game-detail')

    class Meta:
        model = GameCategory
        fields = (
            'url',
            'pk',
            'name',
            'games')


class GameSerializer(serializers.HyperlinkedModelSerializer):
    # We want to display the game cagory's name instead of the id
    owner = serializers.ReadOnlyField(source='owner.username')

    game_category = serializers.SlugRelatedField(queryset=GameCategory.objects.all(), slug_field='name')

    class Meta:
        model = Game
        fields = (
            'url',
            'owner',
            'game_category',
            'name',
            'release_date',
            'played')


class ScoreSerializer(serializers.HyperlinkedModelSerializer):
    # We want to display all the details for the game
    game = GameSerializer()
    # We don't include the player because it will be nested in the player
    class Meta:
        model = PlayerScore
        fields = (
            'url',
            'pk',
            'score',
            'score_date',
            'game',
            )


class PlayerSerializer(serializers.HyperlinkedModelSerializer):
    scores = ScoreSerializer(many=True, read_only=True)
    gender = serializers.ChoiceField(
        choices=Player.GENDER_CHOICES)
    gender_description = serializers.CharField(
        source='get_gender_display', 
        read_only=True)

    class Meta:
        model = Player
        fields = (
            'url',
            'name',
            'gender',
            'gender_description',
            'scores',
            )


class PlayerScoreSerializer(serializers.ModelSerializer):
    player = serializers.SlugRelatedField(queryset=Player.objects.all(), slug_field='name')
    # We want to display the game's name instead of the id
    game = serializers.SlugRelatedField(queryset=Game.objects.all(), slug_field='name')

    class Meta:
        model = PlayerScore
        fields = (
            'url',
            'pk',
            'score',
            'score_date',
            'player',
            'game',
            )


class UserGameSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Game
        fields = (
            'url',
            'name')


class UserSerializer(serializers.HyperlinkedModelSerializer):
    games = UserGameSerializer(many=True, read_only=True)

    class Meta:
        model = User
        fields = (
            'url', 
            'pk',
            'username',
            'games')
