from rest_framework import serializers
from .models import Game


class GameSerializer(serializers.ModelSerializer):
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
implementations for both methods.'''
    class Meta:
        model = Game
        fields = ('id', 
                  'name', 
                  'release_date',
                  'game_category', 
                  'played')
