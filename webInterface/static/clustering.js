
// helper function to get the current target of the event (not the same across browsers)
function target_el(event) {
    var target_el = null;
    if (typeof event.srcElement !== 'undefined') {
        target_el = event.srcElement;
    }
    if (typeof event.target !== 'undefined') {
        target_el = event.target;
    }
    return target_el;
}


// called when we start dragging an element
function dragStartItem(event){

    // allow the element to completely move somewhere else (not get copied)
    event.dataTransfer.effectAllowed = 'move';

    // the data we transfer is the ID of the element; this way we can drop it later
    event.dataTransfer.setData("Text", target_el(event).getAttribute('id'));

    // add class to this element so we can change the style
    target_el(event).classList.add('is-moving');

    // (not sure why we have to return true)
    return true;
}


// called when we stop dragging an element
function dragEndItem(event) {
    // remove class of this element so we can change the style back
    target_el(event).classList.remove('is-moving');
}


// called when we drag an item into a drop area
function dragEnter(event){

    // the item we drag something into
    var target = target_el(event);

    // change style of clusters
    if (target.classList.contains('cluster')) {
        target.classList.add('hover-cluster');
    }

    // if we're hovering over the top cluster, remove top item (if exists)
    if (target.getAttribute('ID') == 'top-cluster') {
        var children = target.children;
        for (var i=0; i<children.length; i++) {
            if (children[i].classList.contains('single-item') && (!children[i].classList.contains('is-moving'))) {
                children[i].classList.add('top-cluster-dragover');
            }
        }
    } else {
        clearCss();
    }

    // tell the event that we can move (not copy) something here if we drop it
    event.dataTransfer.dropEffect = 'move';

    event.preventDefault();
    return true;
}


// called when we drag an item out of a drop area
function dragLeave(event){

    // change style of cluster
    var target = target_el(event);
    if (target.classList.contains('hover-cluster')) {
        target.classList.remove('hover-cluster');
    }

    // if we're hovering over the top cluster, remove top item (if exists)
    if (target.getAttribute('ID') == 'top-cluster') {
        var children = target.children;
        for (var i=0; i<children.length; i++) {
            if (children[i].classList.contains('single-item')) {
                children[i].classList.remove('top-cluster-dragover');
            }
        }
    }

    event.preventDefault();
    return true;
}


// called when we drag an item over another one
function dragOver(event) {

    event.preventDefault();

    // this is the item we are dragging over
    var target = target_el(event);

    // if we drag over an item inside a cluster, we still want that cluster to have a different styling
    var parent = target.parentElement;
    while (!parent.classList.contains('cluster') && parent.parentElement!=null) {
        parent = parent.parentElement;
    }
    if (parent.classList.contains('cluster')) {

        // cluster is still hovered over
        parent.classList.add('hover-cluster');

        // if we're hovering over the top cluster, remove top item (if exists)
        if (parent.getAttribute('ID') == 'top-cluster') {
            var children = parent.children;
            for (var i=0; i<children.length; i++) {
                if (children[i].classList.contains('single-item') && (!children[i].classList.contains('is-moving'))) {
                    children[i].classList.add('top-cluster-dragover');
                }
            }
        }
    }

}


// called when we drop an element
function handleDrop(event){

    clearCss();

    var target = target_el(event);

    // if we drop something in a child of a cluster/unclustered items, go up the chain until we find something where we want to drop it
    while (!target.classList.contains('cluster') && !(target.getAttribute('ID')=='unclustered-items')) {
        target = target.parentElement;
    }

    // make box appear normal again
    target.classList.remove('hover-cluster');

    // if we drop in the top cluster and there was an item in there, move it to second-best cluster
    if (target.getAttribute('ID') == 'top-cluster') {
        var children = target.children;
        for (var i=0; i<children.length; i++) {
            if (children[i].classList.contains('single-item')) {
                // make it appear normal again
                children[i].classList.remove('top-cluster-dragover');
                // put it somewhere else
                var uncl_items = document.getElementById('good-cluster');
                uncl_items.appendChild(children[i]);
            }
        }
    }

    // get the ID of the element we transferred
    var src = event.dataTransfer.getData("Text");
    // get the element we transfer
    var insertedItem = document.getElementById(src);

    // this is only for jobs: we want to change their appearance depending on where we put them
    if (insertedItem.classList.contains('job')) {
        if (target.classList.contains('cluster')) {
            minimise_job(insertedItem);
        } else {
            maximise_job(insertedItem);
        }
    }

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


function clearCss() {
    // if we're hovering over the top cluster, remove top item (if exists)
    var top_items = document.getElementById('top-cluster').children;
    for (var i=0; i<top_items.length; i++) {
        top_items[i].classList.remove('top-cluster-dragover');
    }
}

// make the description of a job tiny
function minimise_job(job) {
    // loop through the items in the job
    var children = job.children;
    for (var i=0; i<children.length; i++) {
        var child = children[i];
        if (child.classList.contains('job-large')) {
            child.classList.add('invisible');
        }
        if (child.classList.contains('job-small')) {
            child.classList.remove('invisible');
        }
    }
}

// make the description of a job large again
function maximise_job(job) {
    var children = job.children;
    for (var i=0; i<children.length; i++) {
        var child = children[i];
        if (child.classList.contains('job-large')) {
            child.classList.remove('invisible');
        }
        if (child.classList.contains('job-small')) {
            child.classList.add('invisible');
        }
    }
}

// called when the user submits the rankng
function handleSubmit() {

    // check if there's a top item
    var top_cluster = document.getElementById('top-cluster');
    var top_item_exists = false;
    for (var i=0; i<top_cluster.children.length; i++) {
        if (top_cluster.children[i].classList.contains('single-item')) {
            top_item_exists = true;
        }
    }
    if (!top_item_exists) {
        window.alert('You have to define a best item, before submitting.');
    }
    // check if there are no more unclustered items
    else if (document.getElementById('unclustered-items').children.length > 0) {
        window.alert('Please sort in all items first.');
    }
    // otherwise save clustering and submit
    else {

        // save the clustering result to the hidden input fields
        save_clustering('top-cluster');
        save_clustering('good-cluster');
        save_clustering('bad-cluster');

        // now submit the form
        document.getElementById("subForm").submit();

    }
}


// helper function for saving the clustering into a hidden field in the submit form
function save_clustering(classname) {
    //get the clustering and save it
    var item_ids = [];
    var items = document.getElementById(classname).children;
    for (var j=0; j < items.length; j++) {
        if (items[j].classList.contains('single-item')) {
            item_ids.push(items[j].getAttribute('id'));
        }
    }
    document.getElementsByName(classname)[0].value = item_ids;
}