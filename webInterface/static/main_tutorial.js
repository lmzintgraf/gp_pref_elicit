function check_pairwise(event, winner, loser) {
    console.log("HALLO", loser, winner);
    if (loser > winner) {
        window.alert('Please click on the option you like better - in this case that means the higher number. In the actual experiment it might not always be as obvious which option is the better one, and you just have to do it to the best of your knowledge.');
    } else {
        return mouseClickJob(event);
    }
}

function check_clustering() {

    // get the clustering
    top_cluster = get_cluster('top-cluster');
    good_cluster = get_cluster('good-cluster');
    bad_cluster = get_cluster('bad-cluster');

    var made_mistake = false;
    best_item = parseInt(top_cluster[0]);
    for (var i=0; i<good_cluster.length; i++) {
        good_item = parseInt(good_cluster[i]);
        if (best_item < good_item) {
            console.log(top_cluster[0], "was smaller than", good_item);
            console.log(top_cluster[0].type);
            made_mistake = true;
        }
        for (var j=0; j<bad_cluster.length; j++) {
            bad_item = parseInt(bad_cluster[j]);
            if (bad_item > good_item) {
                console.log(bad_item, "was larger than", good_item);
                made_mistake = true;
            }
        }
    }

    if (made_mistake) {
        window.alert('Please sort the items so that the best option is in the top cluster, and no item in "good numbers" is smaller than one in "bad numbers". In the actual experiment it may not always be obvious, and it is okay to make mistakes - just answer the questions to the best of your knowledge.');
    } else {
        handleSubmit();
    }
}

function get_cluster(classname) {
        var item_ids = [];
    var items = document.getElementById(classname).children;
    for (var j=0; j < items.length; j++) {
        if (items[j].classList.contains('single-item')) {
            item_ids.push(items[j].getAttribute('id'));
        }
    }
    return item_ids
}

function check_ranking() {

    // get the ranking
    var item_ids = [];
    var ranked_items = document.getElementById("ranked-items").children;
    for (var j=0, item; item = ranked_items[j]; j++) {
        item_ids.push(ranked_items[j].getAttribute('id'));
    }

    var made_mistake = false;
    for (var i=0; i<ranked_items.length-1; i++) {
        if (ranked_items[i].getAttribute('ID') < ranked_items[i+1].getAttribute('ID')) {
            made_mistake = true;
        }
    }

    if (made_mistake) {
        window.alert('Please sort the items so that the best option is on top, and the worst on the bottom. In this case you have to rank the items from highest number to lowest. In the actual experiment it may not always be obvious, and it is okay to make mistakes - just answer the questions to the best of your knowledge.');
    } else {
        handleSubmit();
    }
}