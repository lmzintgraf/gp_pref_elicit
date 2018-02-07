// called when the user submits the ranking
function handleSurveySubmit() {

    // check if all items were moved
    var unclustered = document.getElementById('unclustered-items').children;
    for (var i=0; i<unclustered.length; i++){
        if (unclustered[i].classList.contains('job')) {
            window.alert('Please rank the outcomes first.');
            return;
        }
    }

    var unclustered = document.getElementById('unclustered-preference').children;
    for (var i=0; i<unclustered.length; i++){
        if (unclustered[i].classList.contains('query-type')) {
            window.alert('Please tell us which query type you prefer first.');
            return;
        }
    }

    // check if all multiple choice questions are answered
    var question_names = ['pairwise-effort', 'ranking-effort', 'clustering-effort', 'understand-pairwise', 'understand-ranking', 'understand-clustering'];
    for (var q=0; q<question_names.length; q++) {
        var inputs = document.getElementsByName(question_names[q]);
        var checked = false;
        for (var i=0; i<inputs.length; i++) {
            if (inputs[i].checked) {
                checked = true;
            }
        }
        if (!checked) {
            window.alert("Please answer all questions first :)");
            return;
        }
    }

    // if all of the above passes, save the results in hidden inputs

    // (outcome ranking)
    var outcome_ranking = []
    var cluster_names = ['top-cluster', 'good-cluster', 'bad-cluster']
    for (var j=0; j<3; j++) {
        outcome_ranking.push(document.getElementById(cluster_names[j]).children[1].getAttribute('ID'));
    }
    document.getElementsByName("outcome-ranking")[0].value = outcome_ranking;

    // (outcome preference)
    var preference_ranking = []
    var cluster_names = ['prefer-best', 'prefer-medium', 'prefer-least']
    for (var j=0; j<3; j++) {
        preference_ranking.push(document.getElementById(cluster_names[j]).children[1].getAttribute('ID'));
    }
    document.getElementsByName("preference-ranking")[0].value = preference_ranking;


    // and submit the form
    document.getElementById('subForm').submit();

}













// called when we drag an item over another one
function dragOver(event) {

    event.preventDefault();

    // this is the item we are dragging over
    var target = target_el(event);

    // get the parent cluster if it exists
    var parent = target.parentElement;
    while (!parent.classList.contains('cluster') && parent.parentElement!=null) {
        parent = parent.parentElement;
    }

    if (parent.classList.contains('cluster')) {
        // cluster is still hovered over
        parent.classList.add('hover-cluster');
    }

    // if we drag over a cluster or job inside a cluster, move the job
    if (parent.classList.contains('cluster') || target.classList.contains('cluster')) {
        // if we're hovering over a cluster with a job, move it to the left
        var children = parent.children;
        for (var i=0; i<children.length; i++) {
            if (children[i].classList.contains('job') && (!children[i].classList.contains('is-moving'))) {
                clearCssSurvey();
                children[i].classList.add('cluster-dragover');
            }
        }
    }

    // if we're hovering over an empty cluster, clear css
    if (target.classList.contains('cluster') && target.children.length < 2) {
        clearCssSurvey();
    }

}

// called when we drop an element
function handleDrop(event){

    // clear all extra css we added to anything
    clearCssSurvey();

    var target = target_el(event);

    // if we drop something in a child of a cluster/unclustered items, go up the chain until we find something where we want to drop it
    while (!target.classList.contains('cluster') && !target.classList.contains('unclustered')) {
        target = target.parentElement;
    }

    // make box appear normal again
    target.classList.remove('hover-cluster');

    // if we drop into a cluster and there was an item in there, move it out again
    if (target.classList.contains('cluster')) {
        var children = target.children;
        for (var i=0; i<children.length; i++) {
            if (children[i].classList.contains('job')) {
                // put it somewhere else
                console.log(children[i].classList);
                if (children[i].classList.contains('query-understand')) {
                    var uncl_items = document.getElementById('unclustered-understanding');
                } else if (children[i].classList.contains('query-preference')) {
                    var uncl_items = document.getElementById('unclustered-preference');
                } else {
                    var uncl_items = document.getElementById('unclustered-items');
                }
                uncl_items.appendChild(children[i]);
            }
        }
    }

    // get the ID of the element we transferred
    var src = event.dataTransfer.getData("Text");
    // get the element we transfer
    var insertedItem = document.getElementById(src);

    // put new item into cluster
    target.appendChild(insertedItem);

    // make the dropped item appear differently for a short amount of time
    insertedItem.classList.add('just-dropped');
    setTimeout(function(){ insertedItem.classList.remove('just-dropped'); }, 250);


    // ?
    event.stopPropagation();

    // (note sure why we have to return false here)
    return false;
}

function clearCssSurvey() {
    // if we're hovering over the top cluster, remove top item (if exists)
    var clusters = document.getElementsByClassName('cluster');
    for (var j=0; j<clusters.length; j++) {
        children = clusters[j].children;
        for (var i=0; i<children.length; i++) {
            children[i].classList.remove('cluster-dragover');
        }
    }
}

function checkEnter(event){
    if (event.keyCode == 13) {
        return false;
    }
}